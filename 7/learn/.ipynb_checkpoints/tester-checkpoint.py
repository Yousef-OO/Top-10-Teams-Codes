import torch


class Tester():
  def __init__(self, model, output_hook=None):
    self.model = model
    self.output_hook = output_hook
    self.device = torch.device('cuda')
  
  def test(self, dataloader):
    self.model.eval()
    with torch.no_grad():
      y_scores = torch.empty((0)).float()
      y_probs = torch.empty((0)).float()
      y_preds = torch.empty((0)).float()
      y_trues = torch.empty((0)).float()
      for batch in dataloader:
        waveforms = batch['waveform'].to(self.device)
        y_true = batch['label'].to(self.device)
        output = self.model(waveforms)
        y_score = output[-1] if type(output) is tuple else output
        # unsqueeze target to be of size Bx1
        y_true = y_true.unsqueeze(dim=1)
        y_prob = torch.sigmoid(y_score)
        y_pred = torch.round(y_prob)
        # concatenate the model outputs, predicted labels and true labels
        y_scores = torch.cat((y_scores, y_score.data.cpu()), 0)
        y_probs = torch.cat((y_probs, y_prob.data.cpu()), 0)
        y_preds = torch.cat((y_preds, y_pred.data.cpu()), 0)
        y_trues = torch.cat((y_trues, y_true.data.cpu()), 0)
    return y_trues.numpy(), y_preds.numpy(), y_scores.numpy(), y_probs.numpy()
