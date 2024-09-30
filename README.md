Grond: A Stealthy Backdoor Attack in Model Parameter Space
## Environment settings


## Generate TUAP
```
python generate_tuap.py --model_path {a clean model checkpoint}
```
An example TUAP is provided at ``` results/targeted_uap-cifar10-ResNet18-Linf-eps8.0 ```

## Train Grond backdoor model
```
python train_backdoor.py
```
A trained Grond backdoor checkpoint is provided at ```results/ResNet18-cifar10-STontuap_backdoor-lr0.01-bs128-wd0.0005-pr0.5-seed0-```
