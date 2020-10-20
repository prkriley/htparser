from dynet import *
from utils import ParseForest, read_conll, write_conll
import utils, time, random
import numpy as np

#NEW LSTM: separate p(z) and p(y)

class IndexableModel(Model):
    def __getitem__(self, key):
        self.IndexableModelIndex = {}
        params = self.parameters_list()
        for p in params:
            self.IndexableModelIndex[p.name()] = p
        lookup_params = self.lookup_parameters_list()
        for lp in lookup_params:
            assert lp.name() not in self.IndexableModelIndex, 'Name {} is both a parameter and lookup_parameter!'.format(lp.name())
            self.IndexableModelIndex[lp.name()] = lp
        key = '/' + key
        if key not in self.IndexableModelIndex:
            print('Available keys: {}'.format(list(self.IndexableModelIndex.keys())))
        return self.IndexableModelIndex[key]

class EasyFirstLSTM:
    def __init__(self, words, pos, rels, w2i, options):
        print('Initializing sample-based LSTM model!')
        random.seed(1)
        self.model = IndexableModel()
        self.trainer = AdamTrainer(self.model)

        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.k = options.window
        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
        self.oracle = options.oracle
        self.layers = options.lstm_layers
        self.wordsCount = words
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.builders = [LSTMBuilder(self.layers, self.ldims, self.ldims, self.model), LSTMBuilder(self.layers, self.ldims, self.ldims, self.model)]

        self.blstmFlag = options.blstmFlag
        if self.blstmFlag:
            self.surfaceBuilders = [LSTMBuilder(self.layers, self.ldims, self.ldims * 0.5, self.model), LSTMBuilder(self.layers, self.ldims, self.ldims * 0.5, self.model)]
        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units

        self.external_embedding = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

	    self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.model.add_lookup_parameters("extrn-lookup", (len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.model["extrn-lookup"].init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

	    print 'Load external embedding. Vector dimensions', self.edim

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1

        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.model.add_lookup_parameters((len(words) + 3, self.wdims), name="word-lookup")
        self.model.add_lookup_parameters((len(pos) + 3, self.pdims), name="pos-lookup")
        self.model.add_lookup_parameters((len(rels), self.rdims), name="rels-lookup")

        self.nnvecs = 2

        self.model.add_parameters((self.ldims, self.wdims + self.pdims + (self.edim if self.external_embedding is not None else 0)), name="word-to-lstm")
        self.model.add_parameters((self.ldims), name="word-to-lstm-bias")
        self.model.add_parameters((self.ldims, self.ldims * self.nnvecs + self.rdims), name="lstm-to-lstm")
        self.model.add_parameters((self.ldims), name="lstm-to-lstm-bias")

        self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2 + 1)), name="hidden-layer")
        #self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2)), name="hidden-layer")
        self.model.add_parameters((self.hidden_units), name="hidden-bias")

        self.model.add_parameters((self.hidden2_units, self.hidden_units), name="hidden2-layer")
        self.model.add_parameters((self.hidden2_units), name="hidden2-bias")

        self.model.add_parameters((2, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units), name="output-layer")
        self.model.add_parameters((2), name="output-bias")

        self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2 + 1)), name="rhidden-layer")
        #self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2)), name="rhidden-layer")
        self.model.add_parameters((self.hidden_units), name="rhidden-bias")

        self.model.add_parameters((self.hidden2_units, self.hidden_units), name="rhidden2-layer")
        self.model.add_parameters((self.hidden2_units), name="rhidden2-bias")

        self.model.add_parameters((2 * (len(self.irels) + 0), self.hidden2_units if self.hidden2_units > 0 else self.hidden_units), name="routput-layer")
        self.model.add_parameters((2 * (len(self.irels) + 0)), name="routput-bias")

        self.model.add_parameters((self.hidden_units, self.ldims * self.nnvecs * ((self.k + 1) * 2 + 1)), name="zhidden-layer")
        self.model.add_parameters((self.hidden_units), name="zhidden-bias")
        self.model.add_parameters((self.hidden2_units, self.hidden_units), name="zhidden2-layer")
        self.model.add_parameters((self.hidden2_units), name="zhidden2-bias")
        self.model.add_parameters((1,self.hidden2_units if self.hidden2_units > 0 else self.hidden_units), name="zoutput-layer")
        self.model.add_parameters((1), name="zoutput-bias")


    def  __getExpr(self, forest, i, train):
        #NOTE(prkriley): these are the MLPs that score attachments
        #TODO(prkriley): add params and calculations to return p(z) score as well
        roots = forest.roots
        nRoots = len(roots)

        #NOTE(prkriley): the if-else self.empty is how they do padding
        if self.builders is None:
            raise NotImplementedError
            #input = concatenate([ concatenate(roots[j].lstms) if j>=0 and j<nRoots else self.empty for j in xrange(i-self.k, i+self.k+2) ])
        else:
            input = concatenate([ concatenate([roots[j].lstms[0].output(), roots[j].lstms[1].output()])
                                  if j>=0 and j<nRoots else self.empty for j in xrange(i-self.k-1, i+self.k+2) ])
            #input = concatenate([ concatenate([roots[j].lstms[0].output(), roots[j].lstms[1].output()])
            #                      if j>=0 and j<nRoots else self.empty for j in xrange(i-self.k, i+self.k+2) ])

        if self.hidden2_units > 0:
            routput = (self.routLayer * self.activation(self.rhid2Bias + self.rhid2Layer * self.activation(self.rhidLayer * input + self.rhidBias)) + self.routBias)
            output = (self.outLayer * self.activation(self.hid2Bias + self.hid2Layer * self.activation(self.hidLayer * input + self.hidBias)) + self.outBias)
            zoutput = (self.zoutLayer * self.activation(self.zhid2Bias + self.zhid2Layer * self.activation(self.zhidLayer * input + self.zhidBias)) + self.zoutBias)
        else:
            routput = (self.routLayer * self.activation(self.rhidLayer * input + self.rhidBias) + self.routBias)
            output = (self.outLayer * self.activation(self.hidLayer * input + self.hidBias) + self.outBias)
            zoutput = (self.zoutLayer * self.activation(self.zhidLayer * input + self.zhidBias) + self.zoutBias)

        return routput, output, zoutput


    def __evaluate(self, forest, train):
        nRoots = len(forest.roots)
        nRels = len(self.irels)
        #NOTE(prkriley): they had the -1 because last one didn't need to look to the right
        #for i in xrange(nRoots - 1):
        for i in xrange(1,nRoots):
            if forest.roots[i].scores is None:
                output, uoutput, zoutput = self.__getExpr(forest, i, train)
                scrs = output.value()
                uscrs = uoutput.value()
                zscrs = zoutput.value()
                forest.roots[i].exprs = [(pick(output, j * 2) + pick(uoutput, 0), pick(output, j * 2 + 1) + pick(uoutput, 1)) for j in xrange(len(self.irels))]
                forest.roots[i].scores = [(scrs[j * 2] + uscrs[0], scrs[j * 2 + 1] + uscrs[1]) for j in xrange(len(self.irels))]
                forest.roots[i].zexpr = zoutput


    def Save(self, filename):
        self.model.save(filename)


    def Load(self, filename):
        self.model.populate(filename)


    def Init(self):
        self.word2lstm = parameter(self.model["word-to-lstm"])
        self.lstm2lstm = parameter(self.model["lstm-to-lstm"])

        self.word2lstmbias = parameter(self.model["word-to-lstm-bias"])
        self.lstm2lstmbias = parameter(self.model["lstm-to-lstm-bias"])

        self.hid2Layer = parameter(self.model["hidden2-layer"])
        self.hidLayer = parameter(self.model["hidden-layer"])
        self.outLayer = parameter(self.model["output-layer"])

        self.hid2Bias = parameter(self.model["hidden2-bias"])
        self.hidBias = parameter(self.model["hidden-bias"])
        self.outBias = parameter(self.model["output-bias"])

        self.rhid2Layer = parameter(self.model["rhidden2-layer"])
        self.rhidLayer = parameter(self.model["rhidden-layer"])
        self.routLayer = parameter(self.model["routput-layer"])

        self.rhid2Bias = parameter(self.model["rhidden2-bias"])
        self.rhidBias = parameter(self.model["rhidden-bias"])
        self.routBias = parameter(self.model["routput-bias"])

        self.zhid2Layer = parameter(self.model["zhidden2-layer"])
        self.zhidLayer = parameter(self.model["zhidden-layer"])
        self.zoutLayer = parameter(self.model["zoutput-layer"])

        self.zhid2Bias = parameter(self.model["zhidden2-bias"])
        self.zhidBias = parameter(self.model["zhidden-bias"])
        self.zoutBias = parameter(self.model["zoutput-bias"])

        evec = lookup(self.model["extrn-lookup"], 1) if self.external_embedding is not None else None
        paddingWordVec = lookup(self.model["word-lookup"], 1)
        paddingPosVec = lookup(self.model["pos-lookup"], 1) if self.pdims > 0 else None

        paddingVec = tanh(self.word2lstm * concatenate(filter(None, [paddingWordVec, paddingPosVec, evec])) + self.word2lstmbias )
	self.empty = (concatenate([self.builders[0].initial_state().add_input(paddingVec).output(), self.builders[1].initial_state().add_input(paddingVec).output()]))


    def getWordEmbeddings(self, forest, train):
        for root in forest.roots:
            c = float(self.wordsCount.get(root.norm, 0))
            root.wordvec = lookup(self.model["word-lookup"], int(self.vocab.get(root.norm, 0)) if not train or (random.random() < (c/(0.25+c))) else 0)
            root.posvec = lookup(self.model["pos-lookup"], int(self.pos[root.pos])) if self.pdims > 0 else None

            if self.external_embedding is not None:
                if root.form in self.external_embedding:
                    root.evec = lookup(self.model["extrn-lookup"], self.extrnd[root.form] )
                elif root.norm in self.external_embedding:
                    root.evec = lookup(self.model["extrn-lookup"], self.extrnd[root.norm] )
                else:
                    root.evec = lookup(self.model["extrn-lookup"], 0)
            else:
                root.evec = None

            root.ivec = (self.word2lstm * concatenate(filter(None, [root.wordvec, root.posvec, root.evec]))) + self.word2lstmbias

        if self.blstmFlag:
            forward  = self.surfaceBuilders[0].initial_state()
            backward = self.surfaceBuilders[1].initial_state()

            for froot, rroot in zip(forest.roots, reversed(forest.roots)):
                forward = forward.add_input( froot.ivec )
                backward = backward.add_input( rroot.ivec )
                froot.fvec = forward.output()
                rroot.bvec = backward.output()
            for root in forest.roots:
                root.vec = concatenate( [root.fvec, root.bvec] )
        else:
            for root in forest.roots:
                root.vec = tanh( root.ivec )


    def Predict(self, conll_path):
        with open(conll_path, 'r') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP, False)):
                print("Sentence: {}".format([e.form for e in sentence]))
                self.Init()
                forest = ParseForest(sentence)
                self.getWordEmbeddings(forest, False)

                for root in forest.roots:
                    root.lstms = [self.builders[0].initial_state().add_input(root.vec),
                                  self.builders[1].initial_state().add_input(root.vec)]

                while len(forest.roots) > 1:

                    self.__evaluate(forest, False)
                    #bestParent, bestChild, bestScore = None, None, float("-inf")
                    #bestIndex, bestOp = None, None
                    roots = forest.roots

                    """
                    for i in xrange(len(forest.roots) - 1):
                        for irel, rel in enumerate(self.irels):
                            for op in xrange(2):
                                if bestScore < roots[i].scores[irel][op] and (i + (1 - op)) > 0:
                                    bestParent, bestChild = i + op, i + (1 - op)
                                    bestScore = roots[i].scores[irel][op]
                                    bestIndex, bestOp = i, op
                                    bestRelation, bestIRelation = rel, irel
                    """

                    ###
                    #TODO(prkriley): z score for ROOT should be impossible
                    z_scores = concatenate([r.zexpr for r in roots[1:]])
                    p_z = softmax(z_scores).npvalue()
                    bestIndex = np.argmax(p_z) + 1
                    print('P(z): {}'.format(p_z))
                    print('Best index: {} ({})'.format(bestIndex, roots[bestIndex].form))
                    #TODO(prkriley): p_y
                    valid_exprs = [val for tup in roots[bestIndex].exprs for val in tup]
                    if bestIndex == len(roots) - 1:
                        valid_exprs = valid_exprs[::2]
                    p_y = softmax(concatenate(valid_exprs))
                    max_y_index = np.argmax(p_y.npvalue())

                    if bestIndex < len(roots) - 1:
                        bestOp = max_y_index % 2
                        bestIRelation = (max_y_index - bestOp) / 2
                    else:
                        bestOp = 0
                        bestIRelation = max_y_index
                    #TODO(prkriley): make sure op is valid
                    bestChild = bestIndex
                    bestParent = bestIndex + [-1,1][bestOp]
                    bestRelation = self.irels[bestIRelation]

                    ###

                    #for j in xrange(max(0, bestIndex - self.k - 1), min(len(forest.roots), bestIndex + self.k + 2)):
                    for j in xrange(max(0, bestIndex - self.k - 2), min(len(forest.roots), bestIndex + self.k + 2)):
                        roots[j].scores = None


                    
                    roots[bestChild].pred_parent_id = forest.roots[bestParent].id
                    roots[bestChild].pred_relation = bestRelation

                    roots[bestParent].lstms[bestOp] = roots[bestParent].lstms[bestOp].add_input((self.activation(self.lstm2lstmbias + self.lstm2lstm *
                        	concatenate([roots[bestChild].lstms[0].output(), lookup(self.model["rels-lookup"], bestIRelation), roots[bestChild].lstms[1].output()]))))

                    forest.Attach(bestParent, bestChild)

                renew_cg()
                yield sentence


    def Train(self, conll_path):
        mloss = 0.0
        errors = 0
        batch = 0
        eloss = 0.0
        #eerrors = 0
        #lerrors = 0
        etotal = 0
        #ltotal = 0
        max_quotient = float("-inf")
        min_quotient = float("inf")
        NUM_SAMPLES = 10

        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP, True))
            random.shuffle(shuffledData)

            errs = []
            #eeloss = 0.0

            self.Init()

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 100 == 0 and iSentence != 0:
                    print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Time', time.time()-start
                    #print 'Processing sentence number:', iSentence, 'Loss:', eloss / etotal, 'Errors:', (float(eerrors)) / etotal, 'Labeled Errors:', (float(lerrors) / etotal) , 'Time', time.time()-start
                    start = time.time()
                    #eerrors = 0
                    eloss = 0.0
                    etotal = 0
                    #lerrors = 0
                    #ltotal = 0
                sample_errs = []
                #print('Sentence: {}'.format(sentence))
                for _ in xrange(NUM_SAMPLES):

                    forest = ParseForest(sentence)
                    self.getWordEmbeddings(forest, True)

                    for root in forest.roots:
                        root.lstms = [self.builders[0].initial_state().add_input(root.vec),
                                  self.builders[1].initial_state().add_input(root.vec)]

                    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}
                    #NOTE(prkriley): above looks like number of children; later gets decremented when gets child

                    #loss = 0
                    log_q_total = 0.0
                    log_p_total = 0.0
                    while len(forest.roots) > 1:
                        #TODO(prkriley): sample the next z
                        self.__evaluate(forest, True) #NOTE(prkriley): this updates scores
                        roots = forest.roots

                        rootsIds = set([root.id for root in roots])


                        def _isValid(i):
                            return (unassigned[roots[i].id] == 0) and ((i > 0 and roots[i].parent_id == roots[i-1].id) or (i < len(roots) - 1 and roots[i].parent_id == roots[i+1].id))
                        valid_zs = [j for j in xrange(1,len(roots)) if _isValid(j)]


                        z_scores = concatenate([r.zexpr for r in roots[1:]])
                        valid_z_scores = concatenate([roots[j].zexpr for j in valid_zs])
                        p_zs = softmax(z_scores)
                        #print("P(z): {}".format(p_zs.npvalue()))
                        q_zs = softmax(valid_z_scores)
                        q_zs_numpy = q_zs.npvalue()
                        q_zs_numpy /= np.sum(q_zs_numpy)

                        valid_i = np.random.choice(len(valid_zs),p=q_zs_numpy)
                        q_z = pick(q_zs, valid_i)
                        i = valid_zs[valid_i]
                        log_q_total += log(q_z).scalar_value()
                        p_z = pick(p_zs, i-1)
                        log_p_total += log(p_z).scalar_value()

                        irel = list(self.irels).index(roots[i].relation)
                        op = 0 if roots[i].parent_id == roots[i-1].id else 1
                        #TODO(prkriley): verify correctness of this index math
                        neglog_p_y = pickneglogsoftmax(concatenate([val for tup in roots[i].exprs for val in tup]), irel*2 + op)
                        #TODO(prkriley): change the softmax if p(z) picked the rightmost; only half the values are correct anyway?

                        neglog_p_z = pickneglogsoftmax(z_scores, i-1)
                        errs.append(neglog_p_y + neglog_p_z)
                        log_p_total -= neglog_p_y.scalar_value()
                        mloss += neglog_p_y.scalar_value()
                        mloss += neglog_p_z.scalar_value()

                        etotal += 1
                        

                        selectedChild = i
                        selectedIndex = i
                        selectedOp = op
                        selectedParent = i + [-1,1][op]
                        selectedIRel = irel

                        #TODO(prkriley): better understand this
                            #I think this is marking which ones need to be updated
                        #for j in xrange(max(0, selectedIndex - self.k - 1), min(len(forest.roots), selectedIndex + self.k + 2)):
                        for j in xrange(max(0, selectedIndex - self.k - 2), min(len(forest.roots), selectedIndex + self.k + 2)):
                            roots[j].scores = None

                        #NOTE(prkriley): counts number of real children that are still gettable
                        unassigned[roots[selectedChild].parent_id] -= 1 

                        #NOTE(prkriley): I think lstms[0] is the right one, [1] is the left...
                        roots[selectedParent].lstms[selectedOp] = roots[selectedParent].lstms[selectedOp].add_input(
                                    self.activation( self.lstm2lstm *
                                        noise(concatenate([roots[selectedChild].lstms[0].output(), lookup(self.model["rels-lookup"], selectedIRel),
                                                           roots[selectedChild].lstms[1].output()]), 0.0) + self.lstm2lstmbias))

                        forest.Attach(selectedParent, selectedChild)

                    """
                    #TODO(prkriley): if we can get a stop_gradient, we can do p/q and multiply by the loss
                        #this avoids having to mess with AdamTrainer implementation
                    if len(errs) > 50.0:
                        eerrs = ((esum(errs)) * (1.0/(float(len(errs)))))
                        scalar_loss = eerrs.scalar_value() #NOTE(prkriley): I suspect that this line is not necessary
                        #TODO(prkriley): get P/Q and call scalar_value(), then multiply by that
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []

                        renew_cg()
                        self.Init()
                    """
                    #END OF SENTENCE
                    #TODO(prkriley): finalize loss, do update, etc
                    eerrs = ((esum(errs)) * (1.0/(float(len(errs))))) #TODO(prkriley): consider removing this division
                    #TODO(prkriley): scale by p/q which is exp(logp-logq)
                    #print("logp: {}; logq: {}".format(log_p_total, log_q_total))
                    pq_quotient = np.exp(log_p_total - log_q_total)
                    scaled_pq_quotient = pq_quotient * 1e2
                    eerrs *= scaled_pq_quotient
                    #print("P/Q: {}".format(pq_quotient))
                    max_quotient = max(pq_quotient, max_quotient)
                    min_quotient = min(pq_quotient, min_quotient)
                    eloss += eerrs.scalar_value()
                    sample_errs.append(eerrs)
                    #eerrs.backward()
                    #self.trainer.update()
                    errs = []

                    #renew_cg()
                    #self.Init()
                #END OF SAMPLE
                final_error = esum(sample_errs)
                final_error.backward()
                self.trainer.update()

                renew_cg()
                self.Init()
            #END OF EPOCH
        #FILE CLOSE

        print("Max Quotient: {}; Min Quotient: {}".format(max_quotient, min_quotient))
        if len(errs) > 0:
            eerrs = (esum(errs)) * (1.0/(float(len(errs))))
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()

            errs = []
            lerrs = []

            renew_cg()

        #self.trainer.update_epoch() #TODO(prkriley): verify that AdamTrainer handles everything this did before
        print "Loss: ", mloss/iSentence
