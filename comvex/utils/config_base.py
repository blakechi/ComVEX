class ConfigBase(object):
    def __len__(self) -> int:
        return self.__class__.__dict__.__len__()