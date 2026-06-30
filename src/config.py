CATALOG_PATH = 'data/catalog/books_with_text_image_embeddings_norm.parquet'
TEST_IMAGE_PATH = 'data/test/'

def get_test_image_path(image_number):
    image_basename = f"{image_number:02d}_portada_libro_prueba.jpg"
    return TEST_IMAGE_PATH + image_basename