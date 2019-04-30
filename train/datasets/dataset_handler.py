from glob import glob
import os
from abc import ABC, abstractmethod
import pandas
import re

import helpers

class DatasetHandler(ABC):

    def __init__(self, base_folder, split):
        self.base_folder = base_folder
        self.split = split
        assert split in ["test", "val", "train"], "split '" + split + "' is not valid"

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass
    
    @abstractmethod
    def __len__(self):
        pass

class TessHandler(DatasetHandler):
    #https://tspace.library.utoronto.ca/handle/1807/24487

    def __init__(self, base_folder, split):
        super().__init__(base_folder, split)
        #only one split for this dataset, no need to use it

        self.all_files = glob(os.path.join(base_folder, "*.wav"))
        self.i = 0
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.all_files):
            raise StopIteration
        cur_file = self.all_files[self.i]
        self.i += 1

        emo_id = None
        for emo in helpers.emotion_to_id:
            if emo in cur_file:
                emo_id = helpers.emotion_to_id[emo]
                break
        
        with open(cur_file, "rb") as f:
            data = f.read()

        return {helpers.DATA_KEY: data,
                helpers.EMO_KEY: emo_id,
                helpers.SENT_KEY: 0,
                helpers.EMO_WEIGHT_KEY: 1.,
                helpers.SENT_WEIGHT_KEY: 0.}
    
    def __len__(self):
        return len(self.all_files)

meld_settings = {
    "test": {
        "csv": "test_sent_emo.csv",
        "folder": "output_repeated_splits_test",
    },
    "train": {
        "csv": "train_sent_emo.csv",
        "folder": "train_splits",
    },
    "val": {
        "csv": "dev_sent_emo.csv",
        "folder": "dev_splits_complete",
    },
}

#headers: Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime
class MeldHandler(DatasetHandler):
    #https://github.com/SenticNet/MELD
    def __init__(self, base_folder, split):
        super().__init__(base_folder, split)

        self.settings = meld_settings[split]
        self.csv_file = os.path.join(base_folder, self.settings["csv"])
        self.data_folder = os.path.join(base_folder, self.settings["folder"])

        self.csv_data = pandas.read_csv(self.csv_file)

        self.i = 0
    
    def __iter__(self):
        return self

    def _get_next(self):
        cur_row = self.csv_data.ix[self.i]
        self.i += 1

        emo_id = helpers.emotion_to_id[cur_row.Emotion]
        sent_id = helpers.sentiment_to_id[cur_row.Sentiment]

        dia_id = str(cur_row.Dialogue_ID)
        uter_id = str(cur_row.Utterance_ID)

        file_name = "dia" + dia_id + "_utt" + uter_id + ".wav"
        file_path = os.path.join(self.data_folder, file_name)

        with open(file_path, "rb") as f:
            data = f.read()

        return {helpers.DATA_KEY: data,
                helpers.EMO_KEY: emo_id,
                helpers.SENT_KEY: sent_id,
                helpers.EMO_WEIGHT_KEY: 1.,
                helpers.SENT_WEIGHT_KEY: 1.}

    def __next__(self):
        if self.i >= len(self.csv_data):
            raise StopIteration
        
        ret = None
        while ret is None:
            try:
                ret = self._get_next()
            except Exception as ex:
                print("ERROR:", str(ex))
        
        return ret
    
    def __len__(self):
        return len(self.csv_data)


class SaveeHandler(DatasetHandler):
    #http://kahlan.eps.surrey.ac.uk/savee/
    def __init__(self, base_folder, split):
        super().__init__(base_folder, split)
        
        if split == "train":
            self.all_files = glob(os.path.join(base_folder, "DC", "*.wav"))
            self.all_files += glob(os.path.join(base_folder, "JE", "*.wav"))
            self.all_files += glob(os.path.join(base_folder, "JK", "*.wav"))
        elif split == "val":
            self.all_files = glob(os.path.join(base_folder, "KL", "*.wav"))
            
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.all_files):
            raise StopIteration
        cur_file = self.all_files[self.i]
        self.i += 1

        file_name = os.path.basename(cur_file)
        match = re.match(r"([a-z]+)\d+\.wav", file_name)
        
        emo_abv = match.groups()[0]
        if emo_abv == "a":
            emo_id = helpers.emotion_to_id["anger"]
        elif emo_abv == "d":
            emo_id = helpers.emotion_to_id["disgust"]
        elif emo_abv == "f":
            emo_id = helpers.emotion_to_id["fear"]
        elif emo_abv == "h":
            emo_id = helpers.emotion_to_id["happy"]
        elif emo_abv == "n":
            emo_id = helpers.emotion_to_id["neutral"]
        elif emo_abv == "sa":
            emo_id = helpers.emotion_to_id["sad"]
        elif emo_abv == "su":
            emo_id = helpers.emotion_to_id["surprise"]
        else:
            raise RuntimeError(emo_abv + " is not a valid name")
        
        with open(cur_file, "rb") as f:
            data = f.read()

        return {helpers.DATA_KEY: data,
                helpers.EMO_KEY: emo_id,
                helpers.SENT_KEY: 0,
                helpers.EMO_WEIGHT_KEY: 1.,
                helpers.SENT_WEIGHT_KEY: 0.}
    
    def __len__(self):
        return len(self.all_files)

class CremaHandler(DatasetHandler):
    #https://github.com/CheyneyComputerScience/CREMA-D
    def __init__(self, base_folder, split):
        super().__init__(base_folder, split)
        self.all_files = glob(os.path.join(base_folder, "*.wav"))
            
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= len(self.all_files):
            raise StopIteration
        cur_file = self.all_files[self.i]
        self.i += 1

        file_name = os.path.basename(cur_file)
        match = re.match(r".+_.+_([A-Z]+)_.+.wav", file_name)
        
        emo_abv = match.groups()[0]
        if emo_abv == "ANG":
            emo_id = helpers.emotion_to_id["anger"]
        elif emo_abv == "DIS":
            emo_id = helpers.emotion_to_id["disgust"]
        elif emo_abv == "FEA":
            emo_id = helpers.emotion_to_id["fear"]
        elif emo_abv == "HAP":
            emo_id = helpers.emotion_to_id["happy"]
        elif emo_abv == "NEU":
            emo_id = helpers.emotion_to_id["neutral"]
        elif emo_abv == "SAD":
            emo_id = helpers.emotion_to_id["sad"]
        else:
            raise RuntimeError(emo_abv + " is not a valid name")
        
        with open(cur_file, "rb") as f:
            data = f.read()

        return {helpers.DATA_KEY: data,
                helpers.EMO_KEY: emo_id,
                helpers.SENT_KEY: 0,
                helpers.EMO_WEIGHT_KEY: 1.,
                helpers.SENT_WEIGHT_KEY: 0.}
    
    def __len__(self):
        return len(self.all_files)

