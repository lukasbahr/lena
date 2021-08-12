import yaml
from pathlib import Path
import pprint

class Params():
    def __init__(self, file, args):
        # Load params for specified experiment
        self.params = {}
        load = yaml.load(file, Loader=yaml.FullLoader)
        for d in load:
            self.params.update(d)
        path = Path(args['exp_config']).parent.absolute()
        self.params.update({'path': str(path)})

        if 'tensoboard_path' not in self.params['model']:
            self.params['model']['tensoboard_path'] = str(path) + '/run'

        experiment = self.params['data']['experiment']
        self.params['model']['experiment'] = experiment
        self.params['validation']['experiment'] = experiment

        pprint.pprint(self.params)
        
        # TODO add some validation
