from fastai.text import *

def pg_collate(samples:BatchSamples, pad_idx:int=1, pad_first:bool=True, backwards:bool=False) -> Tuple[LongTensor, LongTensor]:
    "Function that collect samples and adds padding."
    def pad(samples, pad_idx, pad_first, max_len=None, bs=None):
        if not max_len:
            max_len = max([len(s) for s in samples])
        res = torch.zeros(bs, max_len).long() + pad_idx
        for i,s in enumerate(samples):
            if pad_first: res[i,-len(s):] = LongTensor(s)
            else:         res[i,:len(s):] = LongTensor(s)
        return res

    samples = to_data(samples)

    bs = len(samples)

    inp_raw = [s[0][2] for s in samples]
    inp_raw_max_len = max([len(s) for s in inp_raw])

    enc_padding_mask = np.zeros((bs, inp_raw_max_len), dtype=np.float64)
    for i, ex in enumerate(inp_raw):
        ex_len = len(ex)
        # inp_lens.append(ex_len) # TODO: check in evaluation
        for j in range(ex_len):
            enc_padding_mask[i][j] = 1

    enc_batch = pad(inp_raw, pad_idx, pad_first, max_len=inp_raw_max_len, bs=bs)

    enc_batch_extend_vocab = pad([s[0][0] for s in samples], pad_idx, pad_first, bs=bs)

    extra_zeros = None
    max_art_oovs = max([len(article_oovs) for article_oovs in [s[0][3] for s in samples]])
    if max_art_oovs >= 0:
        extra_zeros = torch.zeros(bs, max_art_oovs)

    ids = tensor(np.array([s[0][1] for s in samples], dtype=np.int64))

    has_output = type(samples[0][1]) != int
    if has_output: out = [s[1][0] for s in samples]
    else: out = [s[1] for s in samples]

    dec_padding_mask = np.zeros((bs, max_len), dtype=np.float64)
    dec_lens = np.zeros((bs), dtype=np.int64)
    out_np = np.zeros((bs), dtype=np.int64)

    if has_output:
        for i, ex in enumerate(out):
          ex_len = len(ex)
          # out_lens.append(ex_len) # TODO: check in evaluation
          dec_lens[i] = ex_len
          for j in range(dec_lens[i]):
              dec_padding_mask[i][j] = 1
    else:
        for i, ex in enumerate(out):
            out_np[i] = ex
    
    enc_padding_mask = torch.from_numpy(enc_padding_mask).float()
    dec_padding_mask = torch.from_numpy(dec_padding_mask).float()
    dec_lens = torch.from_numpy(dec_lens).float()
    if not has_output: out = torch.from_numpy(out_np).float()

    inp = [enc_batch_extend_vocab, ids, enc_batch, extra_zeros, enc_padding_mask, dec_padding_mask, dec_lens]
    
    if has_output: out = pad(out, pad_idx, pad_first, max_len=max_len, bs=bs)
    out = [out, ids]
    return inp, out

class PGDataBunch(TextDataBunch):
    "Create a `TextDataBunch` suitable for training an RNN classifier."
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=32, val_bs:int=None, pad_idx=1,
               dl_tfms=None, pad_first=False, device:torch.device=None, no_check:bool=False, backwards:bool=False, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pg_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs//2)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, collate_fn=collate_fn, no_check=no_check)
    
class Text(ItemBase):
    "Basic item for <code>text</code> data in numericalized `ids`."
    def __init__(self, ids, text): self.data,self.text = ids,text
    def __str__(self):  return str(self.text)
    
class PGTextList(TextList):
    _bunch = PGDataBunch
    _label_cls = TextList
    
    def get(self, i):
        o = self.items[i]
        nums = o[0]; idx  = o[1]
        return o if self.vocab is None else Text(o, self.vocab.textify(nums, self.sep, idx=idx))
    
    def reconstruct(self, t:Tensor):
        idx = t[1]; t = t[0]
        idx_min = (t != self.pad_idx).nonzero().min()
        idx_max = (t != self.pad_idx).nonzero().max()
        return Text(t[idx_min:idx_max+1], self.vocab.textify(t[idx_min:idx_max+1], idx=idx))
    
class NumericalizePGInputProcessor(NumericalizeProcessor):
    def process_one(self,item): return [item[0], item[1], np.array(self.vocab.numericalize(item[2]), dtype=np.int64), item[3]]
    def process(self, ds):
        if self.vocab is None: self.vocab = Vocab.create(ds.items, self.max_vocab, self.min_freq)
        else: self.vocab = deepcopy(self.vocab)
        ds.vocab = self.vocab
        self._prepare_input(ds)
        ds.oovs = [item[2] for item in ds.items]
        super().process(ds)

    def _prepare_input(self, ds):
        items = []; oovsl = []; vocab_size = len(self.vocab.itos)
        for idx, item in enumerate(ds.items):
            ids = []; oovs = []
            for w in item:
                i = self.vocab.stoi[w]
                if self.vocab.itos[i] == UNK:
                    if w not in oovs: oovs.append(w) # Add to list of OOVs
                    ids.append(vocab_size + oovs.index(w))  # If w is OOV, This is e.g. 60000 for the first article OOV, 60001 for the second...
                else: ids.append(i)
            oovsl.append(oovs)
            items.append([ids, idx, item, oovs])            
        ds.items = items
        ds.vocab.oovs = oovsl
        
class NumericalizePGOutputProcessor(NumericalizeProcessor):
    def process_one(self,item): return item
    def process(self, ds):
        if self.vocab is None: self.vocab = Vocab.create(ds.items, self.max_vocab, self.min_freq)
        else: self.vocab = deepcopy(self.vocab)
        ds.vocab = self.vocab
        ds.vocab.oovs = ds.x.vocab.oovs
        self._prepare_output(ds)
        super().process(ds)

    def _prepare_output(self, ds):
        items = []; vocab_size = len(self.vocab.itos)
        for idx, (item, oovs) in enumerate(zip(ds.items, ds.vocab.oovs)):
            ids = []
            for w in item:
                i = self.vocab.stoi[w]
                if self.vocab.itos[i] == UNK:
                    if w in oovs: ids.append(vocab_size + oovs.index(w))  # If w is OOV, This is e.g. 60000 for the first article OOV, 60001 for the second...
                    else: ids.append(self.vocab.stoi[UNK])
                else: ids.append(i)
            items.append([ids, idx])            
        ds.items = items

def get_processor(tokenizer:Tokenizer=None, vocab:Vocab=None, chunksize:int=10000, max_vocab:int=60000,
                  min_freq:int=2, mark_fields:bool=False, include_bos:bool=True, include_eos:bool=False,
                  pg_input=False):
    return [TokenizeProcessor(tokenizer=tokenizer, chunksize=chunksize, 
                              mark_fields=mark_fields, include_bos=include_bos, include_eos=include_eos),
            NumericalizePGInputProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq) if pg_input else
            NumericalizePGOutputProcessor(vocab=vocab, max_vocab=max_vocab, min_freq=min_freq)]

def textify(self, nums, sep=' ', idx=None):
    oovs = self.oovs[idx]
    words = []; vocab_size = len(self.itos)
    for i in nums:
        if i < vocab_size: w = self.itos[i] # might be [UNK]
        else:  # w is OOV
            assert oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            oovs_idx = i - vocab_size
            if oovs_idx < len(oovs): w = oovs[oovs_idx]
            else:  # i doesn't correspond to an article oov
                print(sep.join(words))
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                    i, oovs_idx, len(oovs)))
        words.append(w)
    return sep.join(words)

def __getstate__(self):
    state = {'itos': self.itos}
    if hasattr(self, 'oovs'): state['oovs'] = self.oovs
    return state

def __setstate__(self, state:dict):
    self.itos  = state['itos']
    self.stoi  = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})
    if 'oovs' in state: self.oovs  = state['oovs']

Vocab.textify = textify
Vocab.__getstate__ = __getstate__
Vocab.__setstate__ = __setstate__

def _process(self, xp:PreProcessor=None, yp:PreProcessor=None, name:str=None):
    self.x.process(xp)
    self.y.process(yp)
    return self

LabelList.process = _process

def _reconstruct(self, t:Tensor, x:Tensor=None):
    if len(t[0].size()) == 0: return EmptyLabel()
    return self.x.reconstruct(t,x) if has_arg(self.x.reconstruct, 'x') else self.x.reconstruct(t)

EmptyLabelList.reconstruct = _reconstruct

rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4

def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-rand_unif_init_mag, rand_unif_init_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
    linear.weight.data.normal_(std=trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=trunc_norm_init_std)

def init_wt_normal(wt):
    wt.data.normal_(std=trunc_norm_init_std)

def has_issue(inp):
    assert not torch.isnan(inp).any(), 'nan'
    assert not torch.isinf(inp).any(), 'inf'

class PGRNN(nn.Module):
    def __init__(self, emb_enc, emb_dec, nh, out_sl, nl=2, pad_idx=1, bos_idx=2, eos_idx=3, is_half=False):
        super().__init__()
        self.nl,self.nh,self.out_sl,self.pr_force,self.is_half = nl,nh,out_sl,1,is_half
        self.pad_idx,self.bos_idx,self.eos_idx = pad_idx,bos_idx,eos_idx
        
        self.emb_enc,self.emb_dec = emb_enc,emb_dec
                       
        self.emb_sz_enc,self.emb_sz_dec = emb_enc.embedding_dim,emb_dec.embedding_dim
        self.voc_sz_dec = emb_dec.num_embeddings
                 
        self.emb_enc_drop = nn.Dropout(0)
        
        self.lstm_enc = nn.LSTM(self.emb_sz_enc, nh, num_layers=nl, dropout=0.25, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm_enc)

        self.out_enc = nn.Linear(2*nh, self.emb_sz_dec, bias=False)
        
        self.lstm_dec = nn.LSTM(self.emb_sz_dec, nh, num_layers=nl, dropout=0.1, batch_first=True)
        init_lstm_wt(self.lstm_dec)
        
        self.out_drop  = nn.Dropout(0)
        self.out1_drop = nn.Dropout(0)
        
        self.enc_att = nn.Linear(2*nh, 2*nh, bias=False)
        self.hid_att = nn.Linear(2*nh, 2*nh)
        self.V =  self.init_param(2*nh)
        
        self.reduce_h     = nn.Linear(2*nh, nh)
        init_linear_wt(self.reduce_h)
        
        self.reduce_c     = nn.Linear(2*nh, nh)
        init_linear_wt(self.reduce_c)

        self.x_context    = nn.Linear(2*nh + self.emb_sz_dec, self.emb_sz_dec)
        self.p_gen_linear = nn.Linear(4*nh + self.emb_sz_dec, 1)
        
        
        self.out1 = nn.Linear(3*nh, nh)
        self.out2 = nn.Linear(nh, self.voc_sz_dec)
        init_linear_wt(self.out2)
        
    def encoder(self, bs, inp):
        emb = self.emb_enc_drop(self.emb_enc(inp))
        enc_out, hid = self.lstm_enc(emb)
        enc_out = enc_out.contiguous()
        enc_att = enc_out.view(-1, 2*self.nh)  # B * t_k x 2*hidden_dim
        enc_att = self.enc_att(enc_att)
        return hid, enc_out, enc_att
    
    def reduce_state(self, hidden):
        h, c = hidden # h, c dim = 2 x b x hidden_dim
        h_in = h.transpose(0, 1).contiguous().view(-1, 2*self.nh)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))
        c_in = c.transpose(0, 1).contiguous().view(-1, 2*self.nh)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))
        return (hidden_reduced_h.unsqueeze(0).view(self.nl, -1, self.nh),
                hidden_reduced_c.unsqueeze(0).view(self.nl, -1, self.nh)) # h, c dim = 1 x b x hidden_dim
    
    def decoder(self, dec_inp, ctx_1, hid, enc_att, enc_out, enc_padding_mask, extra_zeros, enc_batch_extend_vocab):
        # concatenate decoder embedding with context (we could have just
        # used the hidden state that came out of the decoder, if we weren't
        # using attention)
        emb = self.emb_dec(dec_inp)
        x = self.x_context(torch.cat([emb, ctx_1], 1))
        outp, hid = self.lstm_dec(x[:,None], hid)
        
        # attention scores and ctx calculation
        h_t, c_t = hid
        bs, t_k, n = list(enc_out.size())
        s_t_hat = torch.cat((h_t[-1], c_t[-1]), 1)
        hid_att = self.hid_att(s_t_hat)
        hid_att = hid_att.unsqueeze(1).expand(bs, t_k, n).contiguous() # B x t_k x 2*hidden_dim
        hid_att = hid_att.view(-1, n)  # B * t_k x 2*hidden_dim
        
        # we have put enc_out and hid through linear layers
        u = torch.tanh(enc_att + hid_att) # B * t_k x 2*hidden_dim
        
        # we want to learn the importance of each time step
        scores = u @ self.V # B * t_k x 1
        scores = scores.view(-1, t_k)  # B x t_k
        attn_wgts = F.softmax(scores, dim=1)*enc_padding_mask 
        normalization_factor = attn_wgts.sum(1, keepdim=True)
        eps = 1e-12
        attn_wgts = attn_wgts / (normalization_factor + eps)
        
        # weighted average of enc_out (which is the output at every time step)
        ctx = (attn_wgts[...,None] * enc_out).sum(1)
        
        
        # pointer generator
        p_gen_input = torch.cat((ctx, s_t_hat, x), 1)  # B x (2*2*hidden_dim + emb_dim)
        p_gen = self.p_gen_linear(p_gen_input)
        p_gen = torch.sigmoid(p_gen)
        
        output = torch.cat((outp.view(-1, self.nh), ctx), 1) # B x hidden_dim * 3
        output = self.out1(self.out_drop(output)) # B x hidden_dim
        
        output = self.out2(self.out1_drop(output)) # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_wgts

        if extra_zeros is not None:
            vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)
        
        final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        return final_dist, hid, ctx
        
    def show(self, nm,v):
        if False: print(f"{nm}={v[nm].shape}")
        
    def forward(self, *inp):
        if len(inp) > 7:
            targ, inp = inp[7], inp[:7]
            targ[targ >= self.emb_dec.num_embeddings] = 0
        else: targ = None
        enc_batch_extend_vocab, ids, inp, extra_zeros, enc_padding_mask, dec_padding_mask, dec_lens = inp
        bs, sl = inp.size()
        hid,enc_out, enc_att = self.encoder(bs, inp)
        hid = self.reduce_state(hid)
        dec_inp = inp.new_zeros(bs).long() + self.bos_idx
        ctx = torch.zeros(bs, 2*self.nh).to(dec_inp.device)
        if self.is_half: ctx = ctx.half()
        
        res = []; outs = []
        for i in range(self.out_sl):
            final_dist, hid, ctx = self.decoder(dec_inp, ctx, hid, enc_att, enc_out, enc_padding_mask, extra_zeros,
                                                        enc_batch_extend_vocab)
            res.append(final_dist)
            dec_inp = final_dist.max(1)[1]
            outs.append(dec_inp)
            done = torch.zeros(bs)
            done[(dec_inp == self.eos_idx).nonzero()] = 1
            if (done==1).all(): break
            if (targ is not None) and (random.random()<self.pr_force):
                if i>=targ.shape[1]: continue
                dec_inp = targ[:,i]
        return torch.stack(outs, dim=1), ids, torch.stack(res, dim=1), dec_padding_mask, dec_lens

    def init_param(self, *sz): return nn.Parameter(torch.randn(sz)/math.sqrt(sz[0]))
    
def pg_loss(out, targ, _, pad_idx=1):
    _, _, out, dec_padding_mask, dec_lens = out; eps = 1e-12
    out_len = out.size()[1]; targ_len = targ.size()[1]
    if targ_len>out_len: out  = F.pad(out,  (0,0,0,int(targ_len-out_len),0,0), value=pad_idx)
    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
    gold_probs = torch.gather(out, 2, targ.unsqueeze(2)).squeeze()
    loss  = -torch.log(gold_probs + eps)
    loss = loss * dec_padding_mask
    loss = torch.sum(loss, 1)
    loss = loss/dec_lens
    loss = torch.mean(loss)
    return loss

def pg_acc(out, targ, _, pad_idx=1):
    bs,targ_len = targ.size()
    _, _, out, dec_padding_mask, dec_lens = out; out_len = out.size()[1]
    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)
    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)
    out = out.argmax(2)
    val = (out==targ).float()
    val = val * dec_padding_mask
    val = torch.sum(val, 1)
    val = val/dec_lens
    val = torch.mean(val)
    return val

class TeacherForcing(LearnerCallback):
    def __init__(self, learn, end_epoch=1, pr_force=None):
        super().__init__(learn)
        self.end_epoch,self.pr_force = end_epoch,pr_force
    
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        return {'last_input': last_input+last_target}
    
    def on_epoch_begin(self, epoch, **kwargs):
        if self.pr_force: self.learn.model.pr_force = self.pr_force
        else: self.learn.model.pr_force = 1 - epoch/self.end_epoch
        
class Beam(object):
    def __init__(self, tokens, log_probs, state, context):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context

    def extend(self, token, log_prob, state, context):
        return Beam(tokens = self.tokens + [token],
                    log_probs = self.log_probs + [log_prob],
                    state = state,
                    context = context)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)

def sort_beams(beams):
    return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

def beam_search(learn, inp, sl, no_unk=False, top_k=8, beam_sz=4, min_dec_steps=-1):
    "Return `sl` using beam search."
    from torch.nn.utils.rnn import pad_sequence
    model = learn.model
    inp = inp[:5]
    enc_batch_extend_vocab_lst, ids, inp_lst, extra_zeros_lst, enc_padding_mask_lst = inp
    bs_lst, sl = inp_lst.size()
    bs = 1

    batch_results = []

    for i in range(bs_lst):
        enc_batch_extend_vocab, inp, extra_zeros, enc_padding_mask = \
            enc_batch_extend_vocab_lst[i:i+1], inp_lst[i:i+1], extra_zeros_lst[i:i+1], enc_padding_mask_lst[i:i+1]

        with torch.no_grad():
            hid, enc_out, enc_att = model.encoder(bs, inp)
            hid = model.reduce_state(hid)

            dec_h, dec_c = hid # 1 x 2*hidden_size
            dec_h = dec_h.squeeze(1)
            dec_c = dec_c.squeeze(1)

            ctx = torch.zeros(bs, 2*model.nh).to(enc_out.device)
            if model.is_half: ctx = ctx.half()

            #decoder batch preparation, it has beam_size example initially everything is repeated
            beams = [Beam(tokens=[model.bos_idx], log_probs=[0.0], state=(dec_h, dec_c), context = ctx[0])
                      for _ in range(beam_sz)]

            results = []
            steps = 0
            for k in range(sl):
                if not (len(results) < beam_sz): break

                latest_tokens = [h.latest_token for h in beams]
                latest_tokens = [t if t < model.emb_dec.num_embeddings else learn.data.train_ds.y.vocab.stoi[UNK] for t in latest_tokens]
                dec_inp = LongTensor(latest_tokens).to(enc_out.device)

                all_state_h =[]
                all_state_c = []

                all_context = []

                for h in beams:
                    state_h, state_c = h.state
                    all_state_h.append(state_h)
                    all_state_c.append(state_c)

                    all_context.append(h.context)

                hid = (torch.stack(all_state_h, 0).transpose(0,1).contiguous(), torch.stack(all_state_c, 0).transpose(0,1).contiguous())
                ctx = torch.stack(all_context, 0)

                beams_sz = len(beams)
                _, t_k, n = list(enc_out.size())
                enc_out_expanded = enc_out.expand(beams_sz, t_k, n)
                t_k, n = list(enc_att.size())
                enc_att_expanded = enc_att.unsqueeze(0).expand(beam_sz, t_k, n).contiguous().view(-1, n)
                extra_zeros_expanded = extra_zeros.expand(beam_sz, -1)

                final_dist, hid, ctx = model.decoder(dec_inp, ctx, hid, enc_att_expanded, enc_out_expanded, enc_padding_mask, extra_zeros_expanded,
                                                    enc_batch_extend_vocab)
                eps = 1e-12
                log_probs = torch.log(final_dist + eps)
                if no_unk: log_probs[:,learn.data.train_ds.y.vocab.stoi[UNK]] = -float('Inf')
                topk_log_probs, topk_ids = torch.topk(log_probs, top_k)

                dec_h, dec_c = hid
                dec_h = dec_h.squeeze(1)
                dec_c = dec_c.squeeze(1)

                all_beams = []
                num_orig_beams = 1 if steps == 0 else len(beams)
                for i in range(num_orig_beams):
                    h = beams[i]
                    state_i = (dec_h[:,i], dec_c[:,i])
                    context_i = ctx[i]

                    for j in range(top_k):  # for each of the top_k hyps:
                        new_beam = h.extend(token=topk_ids[i, j].item(),
                                            log_prob=topk_log_probs[i, j].item(),
                                            state=state_i,
                                            context=context_i)
                        all_beams.append(new_beam)

                beams = []
                for h in sort_beams(all_beams):
                    if h.latest_token == model.eos_idx:
                        if steps >= min_dec_steps:
                            results.append(h)
                    else:
                        beams.append(h)
                    if len(beams) == beam_sz or len(results) == beam_sz:
                        break

                steps += 1
        
        if len(results) == 0:
            results = beams

        beams_sorted = sort_beams(results)
        output_ids = [int(t) for t in beams_sorted[0].tokens[1:]]
        batch_results.append(output_ids)
    
    batch_results = pad_sequence([LongTensor(out) for out in batch_results], batch_first=True, padding_value=model.pad_idx)
    return batch_results, ids

def get_predictions(learn: Learner, ds_type=DatasetType.Test, sl=28, use_beam_search=False, no_unk=False, top_k=8, beam_sz=4,
                    min_dec_steps=-1, steps_num=None):
    datasets = {
        DatasetType.Train: learn.data.train_ds,
        DatasetType.Valid: learn.data.valid_ds,
        DatasetType.Test:  learn.data.test_ds
    }
    dls = {
        DatasetType.Train: learn.data.fix_dl,
        DatasetType.Valid: learn.data.valid_dl,
        DatasetType.Test:  learn.data.test_dl
    }
    learn.model.eval()
    res = []

    cb_handler = CallbackHandler()
    cb_handler.set_dl(dls[ds_type])
    with torch.no_grad():
        step = 0
        for xb,yb in progress_bar(dls[ds_type]):
            xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            if use_beam_search: o = beam_search(learn, xb, sl, no_unk=no_unk, top_k=top_k, beam_sz=beam_sz, min_dec_steps=min_dec_steps)
            else:           o = learn.model(*xb)
            res.append((o[0].detach().cpu(), o[1].detach().cpu()))
            if steps_num is not None and step == steps_num-1: break
            step += 1
    outs, ids = [F.pad(pred[0],  (0,sl-pred[0].shape[1],0,0), value=learn.model.pad_idx) for pred in res], [pred[1] for pred in res]
    outs = to_float(torch.cat(outs).cpu()); ids  = to_float(torch.cat(ids).cpu())
    preds = [datasets[ds_type].y.reconstruct([out[:(out == learn.model.eos_idx).nonzero().min()] if learn.model.eos_idx in out else out,i])
             for out,i in zip(outs, ids)]
    return preds, ids

def get_results(preds, df_test, input_col, outout_col, ids=None):
    if ids is not None: df_test = df_test.iloc[ids]
    return [(inp, targ, re.sub(r' +', ' ', re.sub(r'xx[a-zA-Z]+', '', out.text)).strip())
            for inp, targ, out in zip(df_test[input_col], df_test[outout_col], preds)]

def prepare_vocab(df, input_col, output_col, tokenizer=Tokenizer(lang='ar'), chunksize=10000, max_vocab=60000, min_freq=2,
                  mark_fields=False, include_bos=True, include_eos=False):
    inp_ds = TextList.from_df(df, cols=input_col).split_from_df(col='is_valid').train
    out_ds = TextList.from_df(df, cols=output_col).split_from_df(col='is_valid').train
    ds = inp_ds.add(out_ds)
    TokenizeProcessor(tokenizer=tokenizer).process(ds)    
    vocab = Vocab.create(ds.items, max_vocab=max_vocab, min_freq=min_freq)
    return vocab

def prepare_sp_vocab(df, input_col, output_col, path='.', lang='ar', max_vocab=60000, mark_fields=False,
                  include_bos=True, include_eos=False):
    inp_ds = TextList.from_df(df, path=path, cols=input_col).split_from_df(col='is_valid').train
    out_ds = TextList.from_df(df, path=path, cols=output_col).split_from_df(col='is_valid').train
    ds = inp_ds.add(out_ds)
    SPProcessor(max_vocab_sz=max_vocab, lang=lang, mark_fields=mark_fields, include_bos=include_bos, include_eos=include_eos).process(ds)
    return ds.vocab

def _encode_batch(self, texts):
    from sentencepiece import SentencePieceProcessor
    tok = SentencePieceProcessor()
    tok.Load(str(self.sp_model))
    return [np.array(tok.EncodeAsPieces(t)) for t in texts]

SPProcessor._encode_batch = _encode_batch

def get_sp_processor(vocab:Vocab=None, include_eos=False, pg_input=False, path='.'):
    sp = SPProcessor.load(path); sp.include_eos = include_eos
    return [sp, NumericalizePGInputProcessor(vocab=vocab) if pg_input else NumericalizePGOutputProcessor(vocab=vocab)]

def create_emb(vecs, itos, em_sz=300, mult=1., remove=''):
    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)
    wgts = emb.weight.data
    miss = []
    for i,w in enumerate(itos):
        try: wgts[i] = tensor(vecs.get_word_vector(w.replace(remove,'')))
        except: miss.append(w)
    return emb

def set_max_len(max_len):
    def pg_collate(samples:BatchSamples, pad_idx:int=1, pad_first:bool=True, backwards:bool=False) -> Tuple[LongTensor, LongTensor]:    
        "Function that collect samples and adds padding."
        def pad(samples, pad_idx, pad_first, max_len=None, bs=None):
            if not max_len:
                max_len = max([len(s) for s in samples])
            res = torch.zeros(bs, max_len).long() + pad_idx
            for i,s in enumerate(samples):
                if pad_first: res[i,-len(s):] = LongTensor(s)
                else:         res[i,:len(s):] = LongTensor(s)
            return res

        samples = to_data(samples)
        
        bs = len(samples)
        
        inp_raw = [s[0][2] for s in samples]
        inp_raw_max_len = max([len(s) for s in inp_raw])
        
        enc_padding_mask = np.zeros((bs, inp_raw_max_len), dtype=np.float64)
        for i, ex in enumerate(inp_raw):
            ex_len = len(ex)
            # inp_lens.append(ex_len) # TODO: check in evaluation
            for j in range(ex_len):
                enc_padding_mask[i][j] = 1
        
        enc_batch = pad(inp_raw, pad_idx, pad_first, max_len=inp_raw_max_len, bs=bs)
        
        enc_batch_extend_vocab = pad([s[0][0] for s in samples], pad_idx, pad_first, bs=bs)
        
        extra_zeros = None
        max_art_oovs = max([len(article_oovs) for article_oovs in [s[0][3] for s in samples]])
        if max_art_oovs >= 0:
            extra_zeros = torch.zeros(bs, max_art_oovs)
        
        ids = tensor(np.array([s[0][1] for s in samples], dtype=np.int64))
        
        has_output = type(samples[0][1]) != int
        if has_output: out = [s[1][0] for s in samples]
        else: out = [s[1] for s in samples]
        
        dec_padding_mask = np.zeros((bs, max_len), dtype=np.float64)
        dec_lens = np.zeros((bs), dtype=np.int64)
        out_np = np.zeros((bs), dtype=np.int64)
        
        if has_output:
            for i, ex in enumerate(out):
              ex_len = len(ex)
              # out_lens.append(ex_len) # TODO: check in evaluation
              dec_lens[i] = ex_len
              for j in range(dec_lens[i]):
                  dec_padding_mask[i][j] = 1
        else:
            for i, ex in enumerate(out):
                out_np[i] = ex
        
        enc_padding_mask = torch.from_numpy(enc_padding_mask).float()
        dec_padding_mask = torch.from_numpy(dec_padding_mask).float()
        dec_lens = torch.from_numpy(dec_lens).float()
        if not has_output: out = torch.from_numpy(out_np).float()

        inp = [enc_batch_extend_vocab, ids, enc_batch, extra_zeros, enc_padding_mask, dec_padding_mask, dec_lens]
        
        if has_output: out = pad(out, pad_idx, pad_first, max_len=max_len, bs=bs)
        out = [out, ids]
        return inp, out

    @classmethod
    def _create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=32, val_bs:int=None, pad_idx=1,
                dl_tfms=None, pad_first=False, device:torch.device=None, no_check:bool=False, backwards:bool=False, **dl_kwargs) -> DataBunch:
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pg_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs//2)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, collate_fn=collate_fn, no_check=no_check)

    PGDataBunch.create = _create
