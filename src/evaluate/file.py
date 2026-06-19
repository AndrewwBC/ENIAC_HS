import torch
from sklearn.metrics import classification_report as sklearn_report

def evaluate(model, dataloader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad(): 
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['label'].to(device) 

            logits = model(input_ids, attention_mask)
            
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = sklearn_report(
        all_labels, 
        all_preds, 
        target_names=["Não-HS", "Hate Speech"], 
        output_dict=True,
        zero_division=0
    )

    def print_classification_report(report_dict):
        print(f"\n{'Classe':<20} {'precision':<10} {'recall':<10} {'f1-score':<10} {'support':<10}")
        print("-" * 62)
        
        for key, metrics in report_dict.items():
            if key == 'accuracy':
                print(f"{'accuracy':<20} {'':<10} {'':<10} {metrics:<10.4f} {len(all_labels):<10}")
                continue
                
            print(f"{key:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}")
        print("-" * 62)

    print_classification_report(report)
    return report