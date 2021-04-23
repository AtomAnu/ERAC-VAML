import functools
import os, shutil

import numpy as np

import torch


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        os.makedirs(os.path.join(dir_path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return functools.partial(logging, log_path=os.path.join(dir_path, 'log.txt'))

def save_checkpoint(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))

def slow_update(src_model, tgt_model, speed=0.001):
    for p_src, p_tgt in zip(src_model.parameters(), tgt_model.parameters()):
        p_tgt.data.mul_(1 - speed)
        p_tgt.data.add_(speed * p_src.data)

def prepare_for_bleu(ref, hyp, eos_idx, pad_idx, unk_idx, bsz=None, exclude_unk=False):
    if bsz is None: bsz = ref.size(1)
    ref = ref.data.view(ref.size(0), bsz, -1)[1:].clone()
    ref.masked_fill_(ref.eq(eos_idx), pad_idx)
    if exclude_unk:
        ref.masked_fill_(ref.eq(unk_idx), -1)

    hyp = hyp.data.view(hyp.size(0), bsz, -1)[1:].clone()
    hyp.masked_fill_(hyp.eq(eos_idx), pad_idx)

    return ref.permute(1, 2, 0), hyp.permute(1, 2, 0)

def get_rewards(bleu_metric, hyp, ref, return_bleu=False, scale_reward=True):
    """
        hyp - bsz x nhyp x lhyp
        ref - bsz x nref x lref
    """

    # [bsz x nhyp x lhyp] ==> [lhyp x (bsz*nhyp)]
    inc_bleu = bleu_metric.sent_bleu(hyp, ref, inc=True)
    inc_bleu = inc_bleu.view(-1, inc_bleu.size(-1)).transpose(0, 1)

    # r[t] = bleu[t] - bleu[t-1]
    R = inc_bleu.clone()
    R[1:] -= inc_bleu[:-1]

    if scale_reward:
        ref_len = ref.ne(bleu_metric.pad_idx).float().sum(2)
        R *= ref_len.repeat(1, hyp.size(1)).view(-1)

    # whether to return the final bleu score
    if return_bleu:
        return R, inc_bleu[-1]
    else:
        return R

def get_unsuper_rewards(lm, tokenizer,
                        xlm, bpe, dico, params, cos_sim,
                        vocab, src, hyp, inc_adequacy=False, beta=1):

    R = torch.zeros(hyp.size(2), hyp.size(0))

    for i, (src_idx, hyp_idx) in enumerate(zip(src.permute(1, 0), hyp)):
        prev_fluency = 0
        prev_adequacy = 0

        if inc_adequacy:
            src_sent = vocab['src'].convert_to_sent(src_idx.contiguous().data.cpu().view(-1), exclude=[vocab['src'].pad_idx])

        for j in range(0, hyp_idx.size(2)):

            if j == 0:
                hyp_sent = vocab['tgt'].convert_to_sent(hyp_idx[:,:j+1].contiguous().data.cpu().view(-1), exclude=[vocab['tgt'].pad_idx, vocab['tgt'].eos_idx])
                curr_fluency = calculate_fluency(lm, tokenizer, hyp_sent)
            else:
                curr_fluency = 0
            delta_fluency = curr_fluency - prev_fluency

            if inc_adequacy:
                curr_adequacy = calculate_adequacy(xlm, bpe, dico, params, cos_sim, src_sent, hyp_sent)
                delta_adequacy = curr_adequacy - prev_adequacy

                curr_reward = delta_fluency + beta * delta_adequacy
            else:
                curr_reward = delta_fluency

            R[i, j] = curr_reward
            prev_fluency = curr_fluency
            prev_adequacy = curr_adequacy

    return R

def calculate_fluency(lm, tokenizer, hyp_sent):

    with torch.no_grad():
        tokenize_input = tokenizer.tokenize(hyp_sent)
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss = lm(tensor_input, lm_labels=tensor_input)
        fluency = 1/np.exp(loss.item())
    return fluency

def calculate_adequacy(xlm, bpe, dico, params, cos_sim, src_sent, hyp_sent):

    src_hyp_pair = [src_sent, hyp_sent]
    sentences = bpe.apply(src_hyp_pair)

    # add </s> sentence delimiters
    sentences = [(('</s> %s </s>' % sent.strip()).split()) for sent in sentences]

    bs = len(sentences)
    slen = max([len(sent) for sent in sentences])

    word_ids = torch.LongTensor(slen, bs).fill_(params.pad_index)
    for i in range(len(sentences)):
        sent = torch.LongTensor([dico.index(w) for w in sentences[i]])
        word_ids[:len(sent), i] = sent

    lengths = torch.LongTensor([len(sent) for sent in sentences])

    langs = None

    tensor = xlm('fwd', x=word_ids, lengths=lengths, langs=langs, causal=False).contiguous()

    embeddings = tensor[0]
    en_tensor = embeddings[0].unsqueeze(0)
    de_tensor = embeddings[1].unsqueeze(0)

    adequacy = cos_sim(en_tensor, de_tensor)

    return adequacy

def log_sum_exp(x, dim=-1, keepdim=False):
    max_x = x.max(dim=dim, keepdim=True)[0]
    result = (x - max_x).exp().sum(dim, keepdim=True).log() + max_x
    
    if keepdim:
        return result
    else:
        return result.squeeze(dim)

def ngram_sample(sent, nsample, low, high, pad_idx, max_n=4, concat=True):
    start = 1 if concat else 0
    samples = sent[:,:,None].repeat(1, 1, start+nsample)
    lens = sent.ne(pad_idx).int().sum(0)

    # for each sequence in the batch
    for i in range(sent.size(1)):
        # for each ngram sample of the sequence
        for j in range(start, start+nsample):
            n = np.random.randint(1, min(lens[i] - 1, max_n + 1))
            pos = np.random.randint(1, lens[i] - n)
            samples[pos:pos+n,i,j].random_(low, high)

    return samples

def random_mask(seq, rate):
    mask = seq.new(seq.size()).bernoulli_(rate).byte()

    return mask

def random_corrupt(seq, nsample, low, high, pad_idx, bos_idx, eos_idx, rate=0.2, concat=True):
    seqlen, bsz = seq.size()
    seq_crpt = seq[:,:,None].repeat(1, 1, nsample)

    cnt_crpt = 0
    while cnt_crpt == 0:
        mask = random_mask(seq_crpt, rate)
        mask = mask & seq_crpt.gt(pad_idx) & seq_crpt.ne(eos_idx) & seq_crpt.ne(bos_idx)

        cnt_crpt = mask.int().sum()

    data = seq_crpt.new(cnt_crpt).random_(low, high)

    seq_crpt.masked_scatter_(mask, data)

    if concat:
        return torch.cat([seq[:,:,None], seq_crpt], dim=2)
    else:
        return seq_crpt