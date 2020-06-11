import re
import nltk


class LimpiadorTexto:
    '''
    Mi propósito es eliminar las palabras de parada, contracciones y normalizar un texto.
    Para utilizarme debes instanciarme con las palabras de parada.
    
    API:
        normalizar(critica)
        normalizar_fila(fila)
    Los otros métodos son de uso privado y no deberían ser accedidos externamente.

    Ejemplo de uso:

        limpiador = LimpiadorTexto.LimpiadorTexto(palabras_de_parada)
        texto_normalizado = limpiador.normalizar(texto)
    '''

    def __init__(self, palabras_de_parada):
        self.palabras_de_parada = palabras_de_parada

    def descontraer(self, frase):
        # Contracciones específicas
        frase = re.sub(r"won\'t", "will not", frase)
        frase = re.sub(r"can\'t", "cannot", frase)
        frase = re.sub(r"y\'all", "you all", frase)
        # Contracciones generales
        frase = re.sub(r"n\'t", " not", frase)
        frase = re.sub(r"\'re", " are", frase)
        # La contracción 's también podría ser has
        frase = re.sub(r"\'s", " is", frase)
        # La contracción 'd también podría ser had
        frase = re.sub(r"\'d", " would", frase)
        frase = re.sub(r"\'ll", " will", frase)
        frase = re.sub(r"\'t", " not", frase)
        frase = re.sub(r"\'ve", " have", frase)
        frase = re.sub(r"\'m", " am", frase)
        return frase

    def limpiar_html(self, raw_html):
        reg_limpiar = re.compile(
            '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        texto_sin_html = re.sub(reg_limpiar, '', raw_html)
        return texto_sin_html

    def limpiar_texto(self, texto):
        texto_descontraido = self.descontraer(texto)
        texto_limpio = self.limpiar_html(texto_descontraido)
        return texto_limpio

    def quitar_palabras_parada(self, texto_limpio):
        tokens = nltk.tokenize.casual.casual_tokenize(texto_limpio, "english")
        tokens_normalizados = [
            token.lower() for token in tokens if token.lower() not in self.palabras_de_parada]
        return tokens_normalizados

    def normalizar(self, texto):
        texto_limpio = self.limpiar_texto(texto)
        tokens_normalizados = self.quitar_palabras_parada(texto_limpio)
        return " ".join(tokens_normalizados)

    def normalizar_fila(self, fila):
        critica_normalizada =  self.normalizar(fila['Review'])
        fila['Review'] = critica_normalizada
        return fila
