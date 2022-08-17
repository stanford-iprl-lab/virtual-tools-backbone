from typing import Tuple, Annotated, Dict



def word_to_color(colorname: str):
    if colorname is None:
        return None
    try:
        cvec = [int(c) for c in colorname]
        return cvec
    except:
        if colorname == 0:
            return (0, 0, 0, 255)
        c = colorname.lower()
        if c == 'blue':
            return (0,0,255,255)
        elif c == 'red':
            return (255,0,0,255)
        elif c == 'green':
            return (0,255,0,255)
        elif c == 'black':
            return (0,0,0,255)
        elif c == 'white':
            return (255,255,255,255)
        elif c == 'grey' or c == 'gray':
            return (127,127,127,255)
        elif c == 'lightgrey':
            return (191,191,191,255)
        elif c == 'none':
            return (0, 0, 0, 0)
        else:
            raise Exception('Color name not known: ' + c)
