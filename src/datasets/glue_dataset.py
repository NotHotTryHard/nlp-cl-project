from torch.utils.data import Dataset as TorchDataset
from src.datasets import HuggingFaceDataset

class GLUEHuggingFaceDataset(TorchDataset):
    def __init__( 
            self,
            name="mnli",
            streaming=False,
            split=None,
            shuffle=None,
            shuffle_seed=None,
            max_samples=None,
            model_type="enc-dec",
            **kwargs
    ):
        super().__init__()
        self.dataset = HuggingFaceDataset(
            path="nyu-mll/glue",
            name=name,
            streaming=streaming,
            split=split,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            max_samples=max_samples
        )

        self.task_name = name
        self.model_type = model_type

        self.label_mappings = {
            "ax": {0: "entailment", 1: "neutral", 2: "contradiction"},
            "cola": {0: "unacceptable", 1: "acceptable"},
            "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
            "mnli_matched": {0: "entailment", 1: "neutral", 2: "contradiction"},
            "mnli_mismatched": {0: "entailment", 1: "neutral", 2: "contradiction"},
            "mrpc": {0: "not_equivalent", 1: "equivalent"},
            "qnli": {0: "entailment", 1: "not_entailment"},
            "qqp":  {0: "not_duplicate", 1: "duplicate"},
            "rte":  {0: "entailment", 1: "not_entailment"},
            "sst2": {0: "negative", 1: "positive"},
            "wnli": {0: "not_entailment", 1: "entailment"},
        }
        self._preprocess()
    
    def __len__(self):
        return len(self.dataset)

    def map_sample_input(self, sample, prefix):
        if self.task_name in ["ax", "mnli", "mnli_matched", "mnli_mismatched"]:
            inp = prefix + sample["premise"] + " hypothesis: " + sample["hypothesis"]
        elif self.task_name in ["cola", "sst2"]:
            inp = prefix + sample["sentence"]
        elif self.task_name in ["mrpc", "rte", "stsb", "wnli"]:
            inp = prefix + sample["sentence1"] + " sentence2: " + sample["sentence2"]
        elif self.task_name == "qqp":
            inp = prefix + sample["question1"] + " question2: " + sample["question2"]
        elif self.task_name == "qnli":
            inp = prefix + sample["question"] + " sentence: " + sample["sentence"]
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

        if self.model_type == "dec":
            inp = inp + " label:"
        return inp
    
    def _preprocess(self):
        items = []

        for sample in self.dataset:
            prefix = f"{self.task_name}: "
            inputs = self.map_sample_input(sample, prefix)

            if self.task_name == "stsb":
                labels = str(sample["label"])
            else:
                labels = self.label_mappings[self.task_name][sample["label"]]
            
            item = {"text": inputs, "answer": labels, "task_name": self.task_name}
            if self.model_type == "enc-dec":
                item["input"] = inputs
                item["target"] = labels
            else:
                full_text = inputs + labels
                item["input"] = full_text
                item["target"] = full_text
            items.append(item)
        
        self.dataset = items

    def __getitem__(self, idx):
        return self.dataset[idx]
        # input = data["premise"] + " #### " + data["hypothesis"]        
        # target = data["label"]
        # return [input, target]



class SuperGLUEHuggingFaceDataset(TorchDataset):
    def __init__( 
            self,
            name="boolq",
            streaming=False,
            split=None,
            shuffle=None,
            shuffle_seed=None,
            model_type="enc-dec",
            **kwargs
    ):
        """Supported tasks: WiC, CB, COPA, BoolQA, MultiRC, ReCoRD"""

        super().__init__()
        self.dataset = HuggingFaceDataset(
            path="aps/super_glue",
            name=name,
            streaming=streaming,
            split=split,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed
        )

        self.task_name = name
        self.model_type = model_type

        self.label_mappings = {
            "boolqa": {},
            "cb": {0: "unacceptable", 1: "acceptable"},
            "copa": {0: "choice1", 1: "choice2"},
            "multirc": {0: "False", 1: "True"},
            "wic": {0: "False", 1: "True"}
        }
        self._preprocess()
    
    def __len__(self):
        return len(self.dataset)

    def map_sample_input(self, sample, prefix):
        if self.task_name == "cb":
            inp = prefix + sample["premise"] + " hypothesis: " + sample["hypothesis"]
        elif self.task_name == "copa":
            pr, c1, c2, q = sample["premise"], sample["choice1"], sample["choice2"], sample["question"]
            inp = prefix + pr + " question: " + q + " choice1: " + c1 + " choice2: " + c2
        elif self.task_name == "boolq":
            inp = prefix + "passage: " + sample["passage"] + " question: " + sample["question"]
        elif self.task_name == "multirc":
            inp = prefix + "paragraph:" + sample["paragraph"] + " question: " + sample["question"]
        elif self.task_name == "record":
            inp = prefix + "passage: " + sample["passage"] + " question: " + sample["query"]
        elif self.task_name == "wic":
            inp = prefix + "sentence1:" + sample["sentence1"] + " sentence2: " + sample["sentence2"]
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")

        if self.model_type == "dec":
            inp = inp + " label:"
        return inp
    
    def _preprocess(self):
        items = []

        for sample in self.dataset:
            prefix = f"{self.task_name}: "
            inputs = self.map_sample_input(sample, prefix)

            # boolqa, cb, copa, wic - label
            # multirc - answer
            # record - answers[0]

            if self.task_name == "record":
                labels = str(sample["answers"][0])
            elif self.task_name == "multirc":
                labels = self.label_mappings[self.task_name][sample["answer"]]
            else:
                labels = self.label_mappings[self.task_name][sample["label"]]

            item = {"text": inputs, "answer": labels, "task_name": self.task_name}
            if self.model_type == "enc-dec":
                item["input"] = inputs
                item["target"] = labels
            else:
                full_text = inputs + labels
                item["input"] = full_text
                item["target"] = full_text
            items.append(item)

        self.dataset = items

    def __getitem__(self, idx):
        return self.dataset[idx]
        # input = data["premise"] + " #### " + data["hypothesis"]        
        # target = data["label"]
        # return [input, target]