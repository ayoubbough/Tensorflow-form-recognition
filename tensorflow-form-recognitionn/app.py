import numpy as np
import tensorflow as tf

from mnist_utils import DataSets, read_data_sets


def main() -> None:

    try:
        mnist_data: DataSets = read_data_sets()
        x_test: np.ndarray = mnist_data.test.images
        y_test: np.ndarray = mnist_data.test.labels
        total_images: int = x_test.shape[0]
        
        model: tf.saved_model.SavedModel = tf.saved_model.load("./model_keras")
        infer: tf.Tensor = model.signatures["serving_default"]
        
        correct_prediction: int = 0
        
        print(f"Processing {total_images} images...")
        
        # Itération sur le jeu de test
        for i in range(total_images):
            # Préparation du tenseur d'entrée (1, 784)
            input_tensor: tf.Tensor = tf.constant(x_test[i:i+1])
            
            # Exécution de l'inférence
            result: tf.Tensor = infer(input_tensor=input_tensor)
            
            if i == 0:
                print(f"Result keys: {result.keys()}")

            if 'Identity' in result:
                output = result['Identity'].numpy()
            else:
                 output = list(result.values())[0].numpy()
            
            predict: int = int(np.argmax(output))
            
            label: int = int(np.argmax(y_test[i]))
            if predict == label:
                correct_prediction += 1
                
            if (i+1) % 1000 == 0:
                print(f"Processed {i+1}...")
                
        print(
            f"Nous avons {correct_prediction}/{total_images} prédictions correctes "
            f"soit une précision de {correct_prediction/total_images:.2%}."
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

