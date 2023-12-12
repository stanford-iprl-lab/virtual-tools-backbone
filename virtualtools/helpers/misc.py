from typing import Tuple, Annotated, Dict



def word_to_color(colorname: str):
    """Converts a string into (r,g,b,a) color values

    Args:
        colorname (str): The name of the color to translate. Must be in the set of ['blue', 'red', 'green', 'black', 'white', 'grey', 'gray', 'lightgrey', 'none']

    Raises:
        Exception: Raised if the color name is not in the values given above

    Returns:
        Tuple: A length-4 tuple defining an RGBA color
    """    
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
