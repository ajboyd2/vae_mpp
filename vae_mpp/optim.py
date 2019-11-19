import torch
import math

'''
Dictionary of supported optimization algorithms.
'''
OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adadelta": torch.optim.Adadelta,
    "adam": torch.optim.Adam,
    "adagrad": torch.optim.Adagrad,
    "rmsprop": torch.optim.RMSprop,
}


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    '''Learning rate scheduler that is capable of a warm up period, and decaying the learning rate to zero.
    Implementation derived from https://github.com/NVIDIA/Megatron-LM/blob/master/learning_rates.py'''

    # Dictionary of supported learning rate decay schedules.
    # Each value is a function with domains and ranges of: [0, 1] --> [0, 1]
    DECAY_STYLES = {
        "constant": lambda x: 1.0,
        "linear": lambda x: 1.0 - x,
        "cosine": lambda x: 0.5 * (1 + math.cos(math.pi * x)),
        #"exponential": lambda x: pass,
    }

    def __init__(self, optimizer, start_lr, warmup_iter, num_iters, decay_style, args):
        self.optimizer = optimizer
        self.start_lr = float(start_lr)
        self.warmup_iter = warmup_iter
        self.num_iters = 0  # current step
        self.end_iter = num_iters
        self.decay_style = decay_style
        self.decay_func = LRScheduler.DECAY_STYLES[decay_style]

        self.args = args
        if args.loss_monotonic is not None:
            self.loss_annealing = "monotonic"
            self.loss_rate = args.loss_monotonic
        elif args.loss_cyclical is not None:
            self.loss_annealing = "cyclical"
            self.loss_rate = args.loss_cyclical
        else:
            self.loss_annealing = "constant"

        self.true_beta = args.loss_beta
        self.true_lambda = args.loss_lambda

        self.step(self.num_iters)

    def get_lr(self):
        # Ramp up linearly if we are in the warmup period
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return self.start_lr * self.num_iters / self.warmup_iter
        else:
            pct_step = (self.num_iters - self.warmup_iter) / (self.end_iter - self.warmup_iter)
            return self.start_lr * self.decay_func(pct_step)
        
    def get_loss_mult(self):
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            if self.loss_annealing == "constant":
                return 1.0
            else:
                return 0.0
        else:
            adj_iter = self.num_iters - self.warmup_iter
            total_iter = self.end_iter - self.warmup_iter
            pct_step = adj_iter / total_iter
            if self.loss_annealing == "monotonic":
                return min(pct_step / self.loss_rate, 1.0)
            elif self.loss_annealing == "cyclical":
                cycle = int(total_iter * self.loss_rate)
                return min(2.0 * ((adj_iter % cycle) / cycle), 1.0)        
            else:
                return 1.0

    def step(self, step_num=None):
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        mult = self.get_loss_mult() 
       
        self.args.loss_beta = mult * self.true_beta
        self.args.loss_lambda = mult * self.true_lambda

    def state_dict(self):
        sd = {
                'start_lr': self.start_lr,
                'warmup_iter': self.warmup_iter,
                'num_iters': self.num_iters,
                'decay_style': self.decay_style,
                'end_iter': self.end_iter
        }
        return sd

    def load_state_dict(self, sd):
        self.start_lr = sd['start_lr']
        self.warmup_iter = sd['warmup_iter']
        self.num_iters = sd['num_iters']
        self.end_iter = sd['end_iter']
        self.decay_style = sd['decay_style']
        self.decay_func = LRScheduler.DECAY_STYLES[sd['decay_style']]
        self.step(self.num_iters)


def get_optimizer(model, args):

    param_groups = model.get_param_groups()

    optimizer = OPTIMIZERS[args.optimizer](
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,  # learning rate, weight decay, etc.
    )

    return optimizer
    
def get_lr_scheduler(optimizer, args, epoch_len):
    
    total_iterations = args.train_epochs * epoch_len  #  args["train_iters"]
    warmup_iterations = math.floor(args.warmup_pct * total_iterations)

    lr_scheduler = LRScheduler(
        optimizer=optimizer,
        start_lr=args.lr,
        warmup_iter=warmup_iterations,
        num_iters=total_iterations,
        decay_style=args.lr_decay_style,
        args=args,
    )

    return lr_scheduler
