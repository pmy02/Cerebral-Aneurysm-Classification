# Model checkpoints

Trained weights for the four task/circulation models, plus the MedicalNet base.
Large files are tracked with Git LFS (see `../.gitattributes`); run
`git lfs pull` after cloning to fetch them.

| File | Role | Backbone |
|------|------|----------|
| `MedNet_ant_binary.pt`      | Anterior binary detection  | ResNet-18 (MedicalNet) |
| `MedNet_pos_binary.pt`      | Posterior binary detection | ResNet-18 (MedicalNet) |
| `SwinNet_ant_multilabel.pt` | Anterior location          | SwinV2 (timm) |
| `ResNet_pos_multilabel.pt`  | Posterior location         | ResNet-18 (torchvision) |
| `resnet_18.pth`             | MedicalNet base weights    | ResNet-18 |

Place your existing checkpoint files in this directory. To confirm a checkpoint
actually loads into the model architecture, `scripts/evaluate.py` prints a
layer-match summary on load.
