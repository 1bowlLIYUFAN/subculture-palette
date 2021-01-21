# from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel


class Sentence2Vec:
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained("Dee
        # pPavlov/bert-base-multilingual-cased-sentence")
        # self.model = AutoModel.from_pretrained("DeepPavlov/bert-base-multilingual-cased-sentence")
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained("bert-base-multilingual-cased")
        

    def embed(self, input_string):
        # inputs = self.tokenizer(input_string, return_tensors="pt")

        # output = self.model(**inputs)

        encoded_input = self.tokenizer(input_string, return_tensors='pt', padding=True)
        output = self.model(**encoded_input)

        return output


if __name__ == '__main__':
    model = Sentence2Vec()
    result = model.embed(['hello'])

    print(result['pooler_output'].size())
    print(result['last_hidden_state'].size())
