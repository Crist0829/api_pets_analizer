import os
import pika
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

credentials = pika.PlainCredentials('user', 'password')
parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials)

connection = pika.BlockingConnection(parameters)
channel = connection.channel()
channel.queue_declare(queue='image.uploaded', durable=True)
channel.queue_declare(queue='pet.identified', durable=True)

def download_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

model = tf.keras.applications.MobileNetV2(weights="imagenet")

def classify_image(image_url):
    image = download_image(image_url)
    input_array = preprocess_image(image)
    predictions = model.predict(input_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

    print(decoded_predictions)

    labels = [label.lower() for (_, label, _) in decoded_predictions]
    if any("cat" in label for label in labels):
        pet_type = "cat"
    elif any("dog" in label for label in labels):
        pet_type = "dog"
    else:
        return "unknown", None

    breed = decoded_predictions[0][1] if decoded_predictions else "unknown"
    return pet_type, breed

def callback(ch, method, properties, body):
    try:
        message = json.loads(body)
        if 'data' not in message or 'imageUrl' not in message['data']:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        image_url = message['data']['imageUrl']
        pet_type, breed = classify_image(image_url)

        if pet_type == 'unknown':
            print("Unknown pet type. Skipping.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        result_payload = {
            "name": message['data'].get('name'),
            "petId": message['data'].get('petId'),
            "userId": message['data'].get('userId'),
            "imageUrl": image_url,
            "type": pet_type,
            "breed": breed
        }

        channel.basic_publish(
            exchange='',
            routing_key='pet.identified',
            body=json.dumps(result_payload)
        )
        print(f"Sent Data: {result_payload}")

    except Exception as e:
        print(f"Error: {e}")

    ch.basic_ack(delivery_tag=method.delivery_tag)

channel.basic_consume(queue='image.uploaded', on_message_callback=callback)
print('Waiting for messages from RabbitMQ...')
channel.start_consuming()
