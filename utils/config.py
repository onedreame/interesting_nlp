import json

__all__ = ['Config']


class Config(object):
    """Config load from json file
    """

    def __init__(self, config:dict=None, config_file:str=None):
        self.dict = {}
        if config_file:
            with open(config_file, 'r') as fin:
                self.dict.update(json.load(fin))

        if config:
            self.dict.update(config)

        self._update(self.dict)

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, item):
        return item in self.dict

    def items(self):
        return self.dict.items()

    def add(self, key, value):
        """Add key value pair
        """
        self.__dict__[key] = value

    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key])

            if isinstance(config[key], list):
                config[key] = [Config(x) if isinstance(x, dict) else x for x in
                               config[key]]

        self.__dict__.update(config)

if __name__ == '__main__':
    config= Config({'arch': "test", "test": 1}, 'configs/gpt2_chatbot.json')
    print(config.__dict__)
    print(config.arch)
    print(config.test)