import torch 
import torch.nn.functional as F
import numpy as np
from model import MultiDecE2E, E2E
from dataloader import get_data_loader
from dataset import PickleDataset, NegativeDataset
from utils import *
from utils import _seq_mask
import yaml
import os
import pickle

class Solver(object):
    def __init__(self, config, load_model=False):

        self.config = config
        print(self.config)

        # logger
        self.logger = Logger(config['logdir'])

        # load vocab and non lang syms
        self.load_vocab()
       
        # get data loader
        self.get_data_loaders()

        # get label distribution
        self.labeldist = self.get_label_dist(self.train_lab_dataset)

        # calculate proportion between features and characters
        self.proportion = self.calculate_length_proportion()

        # build model and optimizer
        self.build_model(load_model=load_model, multi_dec=self.config['multi_dec'])

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), f'{model_path}.ckpt')
        torch.save(self.gen_opt.state_dict(), f'{model_path}.opt')
        return

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f)
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f)
        return

    def load_model(self, model_path, load_optimizer):
        print(f'Load model from {model_path}.ckpt')
        self.model.load_state_dict(torch.load(f'{model_path}.ckpt'))
        if load_optimizer:
            print(f'Load optmizer from {model_path}.opt')
            self.gen_opt.load_state_dict(torch.load(f'{model_path}.opt'))
        return

    def get_label_dist(self, dataset):
        labelcount = np.zeros(len(self.vocab))
        for _, y in dataset:
            for ind in y:
                labelcount[ind] += 1.
        labelcount[self.vocab['<EOS>']] += len(dataset)
        labelcount[self.vocab['<PAD>']] = 0
        labelcount[self.vocab['<BOS>']] = 0
        labeldist = labelcount / np.sum(labelcount)
        return labeldist

    def calculate_length_proportion(self):
        x_len, y_len = 0, 0
        for x, y in self.train_lab_dataset:
            x_len += x.shape[0]
            y_len += len(y)
        return y_len / x_len
             
    def get_data_loaders(self):
        root_dir = self.config['dataset_root_dir']
        # get labeled dataset
        labeled_set = self.config['labeled_set']
        self.train_lab_dataset = PickleDataset(os.path.join(root_dir, f'{labeled_set}.pkl'), 
            config=self.config, sort=True)
        self.train_lab_loader = get_data_loader(self.train_lab_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'],
                drop_last=False)

        # get unlabeled dataset
        unlabeled_set = self.config['unlabeled_set']
        self.train_unlab_dataset = PickleDataset(
                os.path.join(root_dir, f'{unlabeled_set}.pkl'), 
                config=self.config, sort=True)
        self.train_unlab_loader = get_data_loader(self.train_unlab_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'],
                drop_last=False, speech_only=True)

        # get dev dataset
        dev_set = self.config['dev_set']
        # do not sort dev set
        self.dev_dataset = PickleDataset(os.path.join(root_dir, f'{dev_set}.pkl'), sort=True)
        self.dev_loader = get_data_loader(self.dev_dataset, 
                batch_size=self.config['batch_size'] // 2, 
                shuffle=False, drop_last=False)
        return

    def get_infinite_iter(self):
        # dataloader to cycle iterator 
        self.lab_iter = infinite_iter(self.train_lab_loader)
        self.unlab_iter = infinite_iter(self.train_unlab_loader)
        return

    def build_model(self, load_model=False, multi_dec=True):
        if multi_dec:
            self.model = cc(MultiDecE2E(input_dim=self.config['input_dim'],
                enc_hidden_dim=self.config['enc_hidden_dim'],
                enc_n_layers=self.config['enc_n_layers'],
                subsample=self.config['subsample'],
                vgg=self.config['vgg'],
                dropout_rate=self.config['dropout_rate'],
                dec_hidden_dim=self.config['dec_hidden_dim'],
                att_dim=self.config['att_dim'],
                conv_channels=self.config['conv_channels'],
                conv_kernel_size=self.config['conv_kernel_size'],
                att_odim=self.config['att_odim'],
                output_dim=len(self.vocab),
                embedding_dim=self.config['embedding_dim'],
                ls_weight=self.config['ls_weight'],
                labeldist=self.labeldist,
                pad=self.vocab['<PAD>'],
                bos=self.vocab['<BOS>'],
                eos=self.vocab['<EOS>']
                ))
        else:
            self.model = cc(E2E(input_dim=self.config['input_dim'],
                enc_hidden_dim=self.config['enc_hidden_dim'],
                enc_n_layers=self.config['enc_n_layers'],
                subsample=self.config['subsample'],
                vgg=self.config['vgg'],
                dropout_rate=self.config['dropout_rate'],
                dec_hidden_dim=self.config['dec_hidden_dim'],
                att_dim=self.config['att_dim'],
                conv_channels=self.config['conv_channels'],
                conv_kernel_size=self.config['conv_kernel_size'],
                att_odim=self.config['att_odim'],
                output_dim=len(self.vocab),
                embedding_dim=self.config['embedding_dim'],
                ls_weight=self.config['ls_weight'],
                labeldist=self.labeldist,
                pad=self.vocab['<PAD>'],
                bos=self.vocab['<BOS>'],
                eos=self.vocab['<EOS>']
                ))
        print(self.model)
        self.gen_opt = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], 
                weight_decay=self.config['weight_decay'], amsgrad=True)
        print(self.gen_opt)
        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        return

    def ind2sent(self, all_prediction, all_ys):
        # remove eos and pad
        prediction_til_eos = remove_pad_eos(all_prediction, eos=self.vocab['<EOS>'])

        # indexes to characters
        prediction_sents = to_sents(prediction_til_eos, self.vocab, self.non_lang_syms)
        ground_truth_sents = to_sents(all_ys, self.vocab, self.non_lang_syms)

        # calculate cer
        cer = calculate_cer(prediction_sents, ground_truth_sents)
        return cer, prediction_sents, ground_truth_sents

    def validation(self):
        self.model.eval()
        all_predictions, all_ys = [], []
        total_loss = 0.
        for step, data in enumerate(self.dev_loader):

            xs, ilens, ys = to_gpu(data)

            # calculate loss
            _, log_probs, _, _ = self.model(xs, ilens, ys=ys)

            loss = self.model.mask_and_cal_loss(log_probs, ys)
            total_loss += loss.item()

            # feed previous
            _, _, prediction, _ = self.model(xs, ilens, ys=None, 
                    max_dec_timesteps=self.config['max_dec_timesteps'])

            all_predictions = all_predictions + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.model.train()
        # calculate loss
        avg_loss = total_loss / len(self.dev_loader)
        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_predictions, all_ys)

        return avg_loss, cer, prediction_sents, ground_truth_sents

    def multi_dec_validation(self):
        self.model.eval()
        primal_predictions, aux_predictitions, all_ys = [], [], []
        primal_total_loss, aux_total_loss = 0., 0.
        for step, data in enumerate(self.dev_loader):

            xs, ilens, ys = to_gpu(data)

            # calculate loss
            primal_output, aux_output = self.model(xs, ilens, ys=ys)
            _, log_probs, _, _ = primal_output
            _, aux_log_probs, _, _ = aux_output

            loss = self.model.mask_and_cal_loss(log_probs, ys)
            primal_total_loss += loss.item()
            loss = self.model.mask_and_cal_loss(aux_log_probs, ys)
            aux_total_loss += loss.item()

            # feed previous
            primal_output, aux_output = self.model(xs, ilens, ys=None, 
                    max_dec_timesteps=self.config['max_dec_timesteps'])

            _, _, prediction, _ = primal_output
            _, _, aux_prediction, _ = aux_output

            primal_predictions = primal_predictions + prediction.cpu().numpy().tolist()
            aux_predictions = aux_predictions + aux_prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.model.train()
        # calculate loss
        primal_avg_loss = primal_total_loss / len(self.dev_loader)
        aux_avg_loss = aux_total_loss / len(self.dev_loader)

        cer, prediction_sents, ground_truth_sents = self.ind2sent(primal_predictions, all_ys)

        return primal_avg_loss, aux_avg_loss, cer, prediction_sents, ground_truth_sents

    def test(self, state_dict=None):

        # load model
        if not state_dict:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.model.load_state_dict(state_dict)

        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_set = self.config['test_set']

        test_dataset = PickleDataset(os.path.join(root_dir, f'{test_set}.pkl'), 
            config=None, sort=False)

        test_loader = get_data_loader(test_dataset, 
                batch_size=1, 
                shuffle=False, drop_last=False)

        self.model.eval()
        all_prediction, all_ys = [], []

        for step, data in enumerate(test_loader):

            xs, ilens, ys = to_gpu(data)

            # feed previous
            (_, _ , prediction, _), _ = self.model(xs, ilens, ys=None, 
                    max_dec_timesteps=self.config['max_dec_timesteps'])

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]

        self.model.train()

        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)

        with open(f'{test_set}.txt', 'w') as f:
            for p in prediction_sents:
                f.write(f'{p}\n')

        print(f'{test_set}: {len(prediction_sents)} utterances, CER={cer:.4f}')
        return cer

    def sup_train_one_epoch(self, epoch, tf_rate):

        total_steps = len(self.train_lab_loader)
        total_loss = 0.

        for train_steps, data in enumerate(self.train_lab_loader):

            xs, ilens, ys = to_gpu(data)

            # add gaussian noise after gaussian_epoch
            if self.config['add_gaussian'] and epoch >= self.config['gaussian_epoch']:
                gau = np.random.normal(0, self.config['gaussian_std'], (xs.size(0), xs.size(1), xs.size(2)))
                gau = cc(torch.from_numpy(np.array(gau, dtype=np.float32)))
                xs = xs + gau
            # input the model
            _, log_probs, _, _ = self.model(xs, ilens, ys, tf_rate=tf_rate, sample=False)

            # mask and calculate loss
            loss = -torch.mean(log_probs)
            total_loss += loss.item()

            # calculate gradients 
            self.gen_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
            self.gen_opt.step()
            # print message
            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], '
                f'loss: {loss:.3f}', end='\r')

            # add to logger
            tag = self.config['tag']
            self.logger.scalar_summary(tag=f'{tag}/train_loss', value=loss.item(), 
                    step=epoch * total_steps + train_steps + 1)

        return total_loss / total_steps

    def sup_pretrain(self):
        self.model.train()
        best_cer = 200
        best_model = None
        # tf_rate
        init_tf_rate = self.config['init_tf_rate']
        tf_decay_epochs = self.config['tf_decay_epochs']
        tf_rate_lowerbound = self.config['tf_rate_lowerbound']

        # lr scheduler
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(self.gen_opt, 
        #        milestones=[self.config['change_learning_rate_epoch']],
        #        gamma=self.config['lr_gamma'])

        print('------supervised pretraining-------')
        for epoch in range(self.config['epochs']):
            # schedule
            #scheduler.step()
            # calculate tf rate
            if epoch <= tf_decay_epochs:
                tf_rate = init_tf_rate - (init_tf_rate - tf_rate_lowerbound) * (epoch / tf_decay_epochs)
            else:
                tf_rate = tf_rate_lowerbound

            # train one epoch
            avg_train_loss = self.sup_train_one_epoch(epoch, tf_rate)

            # validation
            avg_val_loss, cer, prediction_sents, ground_truth_sents = self.validation()

            print(f'Epoch: {epoch}, tf_rate={tf_rate:.3f}, train_loss={avg_train_loss:.4f}, '
                    f'valid_loss={avg_val_loss:.4f}, CER={cer:.4f}')

            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/supervised/cer', cer, epoch)
            self.logger.scalar_summary(f'{tag}/supervised/val_loss', avg_val_loss, epoch)
            self.logger.scalar_summary(f'{tag}/supervised/avg_train_loss', avg_train_loss, epoch)

            # only add first n samples
            lead_n = 5
            print('-----------------')
            for i, (p, gt) in enumerate(zip(prediction_sents[:lead_n], ground_truth_sents[:lead_n])):
                self.logger.text_summary(f'{tag}/supervised/prediction-{i}', p, epoch)
                self.logger.text_summary(f'{tag}/supervised/ground_truth-{i}', gt, epoch)
                print(f'hyp-{i+1}: {p}')
                print(f'ref-{i+1}: {gt}')
            print('-----------------')

            if cer < best_cer: 
                # save model 
                model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
                best_cer = cer
                self.save_model(model_path)
                best_model = self.model.state_dict()
                print(f'Save #{epoch} model, val_loss={avg_val_loss:.3f}, CER={cer:.3f}')
                print('-----------------')
            # save model in every epoch
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            self.save_model(f'{model_path}-{epoch:03d}')
        return best_model, best_cer

    def discrepency(self, logits, aux_logits, predictions):
        distr = F.softmax(logits, dim=-1)
        aux_distr = F.softmax(aux_logits, dim=-1)
        # minimizing the discrepency for the two decoders
        discrepency = torch.abs(distr - aux_distr)
        # generate mask by set <EOS> to 0
        mask = (predictions != self.vocab['<EOS>']).float()
        # averaging the output with mask
        dis = torch.sum(discrepency * mask) / torch.sum(mask)
        return dis

    def ssl_train_one_iteration(self, iteration):
        # drawn from labeled and unlabeled data
        lab_data, unlab_data = next(self.lab_iter), next(self.unlab_iter)

        # transfer to GPU
        lab_xs, lab_ilens, lab_ys = to_gpu(lab_data)
        unlab_xs, unlab_ilens = unlab_data
        unlab_xs = cc(unlab_xs)

        # train on source data
        (_, log_probs, _, _), (_, aux_log_probs, _, _) = self.model(lab_xs, lab_ilens, ys=lab_ys, 
                tf_rate=1.0, sample=False, label_smoothing=True)
        primal_loss = -torch.mean(log_probs)
        aux_loss = -torch.mean(aux_log_probs)
        sup_loss = primal_loss + aux_loss
        self.model.zero_grad()
        sup_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
        self.gen_opt.step()

        meta = {'ssl_source/sup_loss': sup_loss.item(),
                'ssl_source/primal_loss': primal_loss.item()}
        # add to logger
        self.logger.scalar_summary(f'{tag}/ssl_source/sup_loss', sup_loss.item(), iteration + 1)
        self.logger.scalar_summary(f'{tag}/ssl_source/primal_loss', primal_loss.item(), iteration + 1)
        self.logger.scalar_summary(f'{tag}/ssl_source/aux_loss', aux_loss.item(), iteration + 1)
        
        # train on the source data, and the decoders to maximize the discrepency
        (_, log_probs, _, _), (_, aux_log_probs, _, _) = self.model(lab_xs, lab_ilens, ys=lab_ys, 
                tf_rate=1.0, sample=False, label_smoothing=True)
        primal_loss = -torch.mean(log_probs)
        aux_loss = -torch.mean(aux_log_probs)
        sup_loss = primal_loss + aux_loss
        (logits, _, predictions, _), (aux_logits, _, _, _) = self.model(unlab_xs, 
            unlab_ilens, ys=None, tf_rate=1.0, sample=self.config['sample'], label_smoothing=False)
        dis_loss = self.discrepency(logits, aux_logits, predictions)
        loss = sup_loss - dis_loss
        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
        self.dec_opt.step()
        # add to logger
        self.logger.scalar_summary(f'{tag}/ssl_decoder/sup_loss', sup_loss.item(), iteration + 1)
        self.logger.scalar_summary(f'{tag}/ssl_decoder/primal_loss', primal_loss.item(), iteration + 1)
        self.logger.scalar_summary(f'{tag}/ssl_decoder/aux_loss', aux_loss.item(), iteration + 1)
        self.logger.scalar_summary(f'{tag}/ssl_decoder/discrepency', dis_loss.item(), iteration + 1)

        # train the encoder to minimize the discrepency
        for enc_step in range(self.config['encoder_iterations']):
            (logits, _, predictions, _), (aux_logits, _, _, _) = self.model(unlab_xs, 
                unlab_ilens, ys=None, tf_rate=1.0, sample=self.config['sample'], label_smoothing=False)
            dis_loss = self.discrepency(logits, aux_logits, predictions)
            self.model.zero_grad()
            dis_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
            self.enc_opt.step()
        # add to logger
        self.logger.scalar_summary(f'{tag}/ssl_encoder/discrepency', dis_loss.item(), iteration + 1)

        return meta 

    def ssl_train(self):
        print('--------SSL training--------')
        # adjust learning rate
        adjust_learning_rate(self.gen_opt, self.config['g_learning_rate'])
        print(self.gen_opt)

        best_cer = 2
        best_model = None

        total_steps = self.config['ssl_iterations']

        if not hasattr(self, 'lab_iter'):
            self.get_infinite_iter()

        for step in range(total_steps):
            meta = self.ssl_train_one_iteration(iteration=step)
            # printed message 
            sup_loss, unsup_loss, loss = meta['sup_loss'], meta['unsup_loss'], meta['loss']
            print(f'[{step + 1}/{total_steps}], sup_loss: {sup_loss:.3f}, unsup_loss: {unsup_loss:.3f}, '
                    f'loss: {loss:.3f}', end='\r')

            if (step + 1) % self.config['summary_steps'] == 0 or step + 1 == total_steps:
                avg_valid_loss, cer, prediction_sents, ground_truth_sents = self.validation()

                print(f'Iter: [{step + 1}/{total_steps}], valid_loss={avg_valid_loss:.4f}, CER={cer:.4f}')

                # add to tensorboard
                tag = self.config['tag']
                self.logger.scalar_summary(f'{tag}/ssl/cer', cer, step + 1)
                self.logger.scalar_summary(f'{tag}/ssl/val_loss', avg_valid_loss, step + 1)

                # only add first n samples
                lead_n = 5
                print('-----------------')
                for i, (p, gt) in enumerate(zip(prediction_sents[:lead_n], ground_truth_sents[:lead_n])):
                    self.logger.text_summary(f'{tag}/ssl/prediction-{i}', p, step + 1)
                    self.logger.text_summary(f'{tag}/ssl/ground_truth-{i}', gt, step + 1)
                    print(f'hyp-{i+1}: {p}')
                    print(f'ref-{i+1}: {gt}')
                print('-----------------')

                if cer < best_cer: 
                    # save model 
                    model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
                    best_cer = cer
                    self.save_model(model_path)
                    self.save_judge(model_path)
                    best_model = self.model.state_dict()
                    print(f'Save #{step} model, val_loss={avg_valid_loss:.3f}, CER={cer:.3f}')
                    print('-----------------')
        
if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    solver = Solver(config)
