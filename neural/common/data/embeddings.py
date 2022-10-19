import shutil
import tarfile
from pathlib import Path

from torchtext.utils import download_from_url
from torchtext.vocab import Vectors


class CollobertEmbeddings(Vectors):
    __EMBEDDING_URL = 'http://ronan.collobert.com/senna/senna-v3.0.tgz'
    __EXTRACTED_DIR = 'senna'
    __WORDS_LIST_PATH = 'hash/words.lst'
    __VECTORS_LIST_PATH = 'embeddings/embeddings.txt'
    __EMBEDDING_FILENAME = 'collobert_embeddings.txt'

    def __init__(self, embedding_dir: str):
        embedding_file = self.__create_embedding_file(Path(embedding_dir))
        super().__init__(embedding_file, embedding_dir)

    def __create_embedding_file(self, embedding_dir: Path) -> Path:
        embedding_dir.mkdir(parents=True, exist_ok=True)
        embedding_file = embedding_dir / self.__EMBEDDING_FILENAME

        if embedding_file.exists():
            return embedding_file

        senna_file = download_from_url(self.__EMBEDDING_URL, root=embedding_dir)
        with tarfile.open(senna_file, 'r') as tar_file:
            
            import os
            
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar_file, embedding_dir)

        words_file = embedding_dir / self.__EXTRACTED_DIR / self.__WORDS_LIST_PATH
        vectors_file = embedding_dir / self.__EXTRACTED_DIR / self.__VECTORS_LIST_PATH

        # Construct embedding file with the same pattern as torchtext
        with open(words_file, 'r') as words, open(vectors_file, 'r') as vectors, open(embedding_file, 'w') as embedding:
            for word, vector in zip(words, vectors):
                embedding.write(f'{word.strip()} {vector.strip()}\n')

        # Remove downloaded files
        Path(senna_file).unlink()
        shutil.rmtree(embedding_dir / self.__EXTRACTED_DIR)

        return embedding_file
