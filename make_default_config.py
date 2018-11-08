import configparser

def get_config():
    """Creates the default configuration file for RetinaChecker
    
    Returns:
        ConfigParser -- default / example configuration file
    """
    config = configparser.ConfigParser()

    config['network'] = {
    'model': 'resnet18',
    'pretrained': False,
    'optimizer': 'Adam',
    'criterion': 'CrossEntropyLoss'
    }

    config['hyperparameter'] = {
    'epochs': 50,
    'batch size': 64,
    'learning rate': 0.001
    }

    config['files'] = {
    'train path': './train',
    'test path': './test',
    'show patch stats': False,
    'image size': 224,
    'samples': 6400
    }

    config['transform'] = {
    'rotation angle': 180,
    'brightness': 0,
    'contrast': 0,
    'saturation': 0,
    'hue': 0
    }

    config['output'] = {
    'save during training': True,
    'save every nth epoch': 1,
    'filename': 'model',
    'extension': '.ckpt',
    'cleanup': True
    }

    config['input'] = {
    'checkpoint': 'model.ckpt',
    'resume': False,
    'evaluation only': False
    }

    return config

def save_default_config( filename = 'default.cfg' ):
    """Stores the default configuration file
    
    Keyword Arguments:
        filename {str} -- target configuration file (default: {'default.cfg'})
    """
    config = get_config()
    with open(filename, 'w') as fopen:
        config.write(fopen)

if __name__ == '__main__':
    save_default_config('default.cfg')