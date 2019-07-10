from difflib import SequenceMatcher
from face_swap_main import face_swapper
import cv2

templates = {'one does not simply': 'memes_templates/onedoesnotsimply.png',
             'brace yourselves': 'memes_templates/braceyourselves.png',
             'cheers': 'memes_templates/cheers.png',
             'is this a': 'memes_templates/isthisa.png',
             'think about it': 'memes_templates/thinkaboutit.png',
             'what the hell':'memes_templates/whatthehell.png',
             'wtf ': 'memes_templates/wtf.png',
             'wat ': 'memes_templates/wat.png',
             'you get a': 'memes_templates/yougeta.png',
             'you mean to tell me': 'memes_templates/youmeantotellme.png',
             'ha-ha':'memes_templates/haha.png',
             'now tell me':'memes_templates/nowtellme.png',
             'are you kidding': 'memes_templates/areyoukidding.png',
             'no time to explain': 'memes_templates/notimetoexplain.png',
             'dont mess': 'memes_templates/dontmess.png',
             'rofl ': 'memes_templates/rofl.png',
             'epic fail ': 'memes_templates/epicfail.png'
             }


def memes_generate(img, msg, out):
    tmpl = get_mem_templates(msg)
    if tmpl is not None:
        face_swapper(img, cv2.imread(tmpl[0]), out=out, text=tmpl[1])


def get_mem_templates(phrase):
    for ph, templ in templates.items():
        r = SequenceMatcher(a=phrase[:len(ph)], b=ph).ratio()
        if r > 0.72:
            return (templ, phrase[len(ph):])
    return None
