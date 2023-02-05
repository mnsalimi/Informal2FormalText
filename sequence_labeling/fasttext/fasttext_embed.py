
import fasttext
# import fasttext.util

class FastText():
    
    def __init__(self):
        self.fasttext = fasttext.load_model('cc.fa.300.bin.gz')

    def get_vector(self, token):
        try:
            embed = self.fasttext.get_word_vector(token)
        except:
            return []
        return embed

if __name__ == "__main__":
    fasttext = FastText()
    embed = fasttext.get_vector()
    print(embed)