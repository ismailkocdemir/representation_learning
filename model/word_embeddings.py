import os, sys
import copy
import torch
import torch.nn as nn
import numpy as np

sys.path.append("..")
from util import io
        
class ViCoWordEmbeddings(nn.Module):
    def __init__(self,
            root,    
            num_classes,
            vico_mode,
            one_hot=False,
            linear_dim = 100,
            no_hypernym=False,
            no_glove=False,
            pool_size=None
        ):
        super(ViCoWordEmbeddings,self).__init__()
        
        self.num_classes = num_classes
        self.pool_size = pool_size
        self.one_hot = one_hot
        if one_hot:
            self.embed = nn.Embedding(
                self.num_classes,
                self.num_classes
                ).requires_grad_(False)
        else:
            self.vico_mode = vico_mode
            self.linear_dim = linear_dim
            self.glove_dim = 300
            self.embed_dims = self.glove_dim

            if self.vico_mode == 'vico_linear':
                self.embed_dims += self.linear_dim
                embed_dir = 'glove_300_vico_linear_{}'.format(self.linear_dim)    
            elif self.vico_mode == 'vico_select':
                self.embed_dims += 200
                embed_dir = 'glove_300_vico_select_200'
            else:
                raise ValueError("Unknown embedding type:{}".format(self.vico_mode))

            
            self.original_embed_dims = self.embed_dims
            
            self.embed_h5py = os.path.join(root, embed_dir, 'visual_word_vecs.h5py')
            self.embed_word_to_idx_json = os.path.join(root, embed_dir, 'visual_word_vecs_idx.json')

            self.no_glove = no_glove
            if no_glove:
                self.embed_dims -= self.glove_dim

            self.no_hypernym = no_hypernym
            if no_hypernym:
                self.embed_dims -= 50

            self.embed = nn.Embedding(
                self.num_classes,
                self.embed_dims).requires_grad_(False)
        

    def load_embeddings(self,labels):
        if self.one_hot:
            self.embed.weight.data.copy_(torch.eye(self.num_classes).cuda())
            return
        else:
            embed_h5py = io.load_h5py_object(self.embed_h5py)['embeddings']
            word_to_idx = io.load_json_object(self.embed_word_to_idx_json)
            embeddings = np.zeros([len(labels),self.original_embed_dims])
            word_to_label = {}
            for i,label in enumerate(labels):
                if ' ' in label:
                    words = label.split(' ')
                elif '_' in label:
                    words = label.split('_')
                else:
                    words = [label]

                denom = len(words)
                for word in words:
                    if word=='tree':
                        denom = len(words)-1
                        continue

                    if word not in word_to_label:
                        word_to_label[word] = set()
                    word_to_label[word].add(label)

                    idx = word_to_idx[word]
                    embeddings[i] += embed_h5py[idx][()]
                embeddings[i] /= denom

            if self.no_glove:
                embeddings = np.delete(embeddings, np.s_[:self.glove_dim], axis=1) 
                #embeddings[:,:self.glove_dim] = 0
            if self.vico_mode == 'vico_select' and self.no_hypernym:
                embeddings = np.delete(embeddings, np.s_[100:150], axis=1)
                #embeddings[:,self.glove_dim+100:self.glove_dim+150] = 0

            self.embed.weight.data.copy_(torch.from_numpy(embeddings))
            #self.embed.weight -= torch.mean(self.embed.weight, dim=0, keepdims=True)

    def forward(self, feats, label_idxs, target):
        feats = self.pool_feats(feats, self.pool_size)
        feats = feats - torch.mean(feats, dim=0, keepdims=True)
        
        embed = torch.index_select(self.embed.weight, 0, label_idxs)
        #embed = torch.index_select(self.embed, 0, label_idxs)
        embed = embed - torch.mean(embed, dim=0, keepdims=True)

        class_sim = self.center_gram(self.gram_linear(feats))
        embed_sim = self.center_gram(self.gram_linear(embed))  
        
        embed_norm = torch.norm(embed_sim, dim=(0,1))
        class_norm = torch.norm(class_sim, dim=(0,1))
        cka = torch.sum((class_sim*embed_sim))/(class_norm*embed_norm)

        """
        DEBUGGING FOR DIFFERENT SIMILARITY TARGETS
        with torch.no_grad():
            cka2 = torch.sum((class_sim/class_norm)*(embed_sim/embed_norm))
            l2_loss = torch.sum(((class_sim/class_norm) - (embed_sim/embed_norm))**2)
            print("Normal CKA", cka)
            print("CKA_v2", cka2)
            print("2-2CKA_v2", 2*(1-cka2))
            print("SymNMF", l2_loss)
        """

        zero_tensor = torch.FloatTensor([0]).to(target.device)
        return cka, torch.max(zero_tensor, target-cka)**2

    def pool_feats(self, feats, pool_size):
        if len(feats.shape) > 2 and self.pool_size:
            feats = nn.functional.adaptive_avg_pool2d(feats, pool_size)
        return feats.reshape(feats.shape[0], -1)

    def gram_linear(self, x):
        return torch.mm(x,x.t())

    def center_gram(self, gram):
        n = gram.shape[0]        
        gram = self.subtract_diag(gram)
        means = torch.sum(gram, dim=0) / (n - 2)
        means -= torch.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        gram = self.subtract_diag(gram)
        return gram

    def subtract_diag(self, gram):
        diag_elements = torch.diag_embed(torch.diagonal(gram, 0))
        gram -= diag_elements
        return gram
