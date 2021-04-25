class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        print(args)
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class ConfigReader(dict):
    def __init__(self, file_path, **kwargs):    

        def convert(val):
            if val == 'None': return None
            if val == 'True': return True
            if val == 'False': return False
            constructors = [int, float, str]
            for c in constructors:
                try:            
                    return c(val)
                except ValueError:
                    pass
            return None

        config_dict = {}    
        with open(file_path, "r") as fp:
            lines = fp.readlines()
            for line in lines:            
                if len(line.strip()) > 0:
                    try:
                        eqi = line.index('=')
                        key = line[:eqi].strip()
                        value = line[eqi + 1:].strip()
                        config_dict[key] = convert(value)
                    except:
                        pass
        # print(config_dict)
        super(ConfigReader, self).__init__(config_dict, **kwargs)
        self.__dict__ = self

    def hasattr(self, attr):
        return attr in self.__dict__

    def getattr(self, attr):
        if attr in self.__dict__:
            return self.__dict__[attr]
        else:
            return None

import matplotlib.pyplot as plt
def plot_images(images, n_rows, n_cols, scale = 5., save_loc = None):
    plt.figure(figsize=(n_cols * scale, n_rows * scale))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            if index >= len(images):
                break
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(images[index])
            plt.axis('off')

    # plt.subplots_adjust(wspace=0.2, hspace=0.2)

    if save_loc is not None:
        plt.savefig(save_loc, tight_layout=True)