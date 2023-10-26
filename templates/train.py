import torch
from lightning.pytorch.demos import WikiText2, Transformer
import lightning as L
import wandb
import argparse
from tqdm import tqdm

def run(params):
  if params.log:
    wandb.init(project=params.project_name)
    for k, v in params.__dict__.items():
      wandb.config[k] = v

  fabric = L.Fabric(accelerator="cuda", devices=params.devices, strategy=params.strategy)
  fabric.launch()

  dataset = WikiText2()
  dataloader = torch.utils.data.DataLoader(dataset, num_workers=2, batch_size=params.batch)
  model = Transformer(vocab_size=dataset.vocab_size)
  optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)

  model, optimizer = fabric.setup(model, optimizer)
  dataloader = fabric.setup_dataloaders(dataloader)

  model.train()
  for epoch in range(params.epochs):
    with tqdm(dataloader, unit="batch") as tepoch:
      for input, target in tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")
        optimizer.zero_grad()
        output = model(input, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        fabric.backward(loss)
        optimizer.step()
        tepoch.set_postfix(loss=loss.item())

    if params.log:
      wandb.log({
        "epoch": epoch,
        "test_loss": loss.item()
      })

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="god tier torch training script")
  parser.add_argument("--batch", type=int, default=64, help="training batch size (default: 64)")
  parser.add_argument("--epochs", type=int, default=20, help="training epochs (default: 20)")
  parser.add_argument("--devices", type=int, default=1, help="num accelerators (default: 1)")
  parser.add_argument("--precision", type=str, default="b16-mixed", help="precision (default: b16-mixed)")
  parser.add_argument("--strategy", type=str, default="dp", help="parallelism strategy (default: dp)")
  parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
  parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
  parser.add_argument("--seed", type=int, default=42, help="random seed (default: 1)")
  parser.add_argument("--log", action="store_true", default=False, help="enable W&B log")
  parser.add_argument("--project-name", type=str, default="", help="W&B project name")
  parser.add_argument("--model-name", type=str, default="", help="ckpt model name")
  parser.add_argument("--save", action="store_true", default=False, help="saving checkpoint for trained model")
  hparams = parser.parse_args()
  run(hparams)
