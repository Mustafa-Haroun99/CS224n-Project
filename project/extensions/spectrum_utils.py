import re

def freeze_model(model, file_name):
    with open(file_name, "r") as fin:
        yaml_parameters = fin.read()
   
    unfrozen_parameters = []
    for line in yaml_parameters.splitlines():
        if line.startswith("- "):
            unfrozen_parameters.append(line.split("- ")[1])

        def freeze_and_unfreeze_parameters(model, unfrozen_parameters):
            # freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            # unfreeze Spectrum parameters
            for name, param in model.named_parameters():
                if any(re.match(unfrozen_param, name) for unfrozen_param in unfrozen_parameters):
                    param.requires_grad = True

    freeze_and_unfreeze_parameters(model, unfrozen_parameters)

if __name__ == '__main__':

    # let's do a quick sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)