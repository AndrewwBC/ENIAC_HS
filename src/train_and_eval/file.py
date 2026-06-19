from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from nerdataset.file import NERDataset
from torch.utils.data import DataLoader
from src.train_one_epoch.file import train_one_epoch
from src.evaluate.file import evaluate
from src.bert.file import BertForTextEmbedding

class TrainAndEval():
    def __init__(self, hparams, train_df, eval_df, device):
        self.device  = device
        self.hparams = hparams
        self.train_df = train_df
        self.eval_df  = eval_df
    
    def dataloader(self):
        train_ds = NERDataset(self.train_df)
        eval_ds  = NERDataset(self.eval_df)
                
        self.train_loader = DataLoader(train_ds, batch_size=self.hparams.get('batch_size'), shuffle=True)
        self.eval_loader  = DataLoader(eval_ds,  batch_size=self.hparams.get('batch_size'), shuffle=False)
    
    def __load_model(self):
        nn_params = self.hparams.get("nn", {})
        
        return BertForTextEmbedding(
            dropout=nn_params.get("dropout")
        ).to(self.device)

    def __optimizer(self):
        bert_params   = self.hparams.get("bert", {})
        linear_params = self.hparams.get("nn", {})

        bert_lr = bert_params.get("learning_rate")
        bert_wd = bert_params.get("weight_decay")

        nn_lr = linear_params.get("learning_rate")
        nn_wd = linear_params.get("weight_decay")

        optimizer = AdamW([
            {
                "params": self.model.bert.parameters(),
                "lr": bert_lr,
                "weight_decay": bert_wd
            },
            {
                "params": self.model.linear.parameters(), 
                "lr": nn_lr, 
                "weight_decay": nn_wd
            }
        ])

        return optimizer
    def __scheduler(self, optimizer):
        total_steps = len(self.train_loader) * self.hparams['epochs']
        
        warmup_ratio = self.hparams.get('warmup_ratio', 0.1)
        num_warmup_steps = int(warmup_ratio * total_steps)
        
        scheduler_type = self.hparams.get('lr_scheduler', 'linear')

        if scheduler_type == 'linear':
            return get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps
            )
            
        elif scheduler_type == 'cosine':
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps
            )
            
        else:
            raise ValueError(f"Scheduler {scheduler_type} não reconhecido.")
    def __call__(self):
        best_acc = 0.0
        best_state = None
        
        self.model = self.__load_model()      
        self.dataloader()
        
        optimizer = self.__optimizer()
        scheduler = self.__scheduler(optimizer)

        for epoch in range(self.hparams['epochs']):
            train_loss = train_one_epoch(
                self.model,
                self.train_loader, 
                self.device, 
                optimizer,
                scheduler, 
            )
            print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
            results = evaluate(self.model, self.eval_loader, self.device)         

            acc = results.get("accuracy")
                        
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v for k, v in self.model.state_dict().items()}
        
        return results, best_state
