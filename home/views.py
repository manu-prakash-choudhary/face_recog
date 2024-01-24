from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from deepface import DeepFace
import cv2
import pickle
from mtcnn import MTCNN
import numpy as np


# Create your views here.
# return Response({"message": "Hello, World!"}) returns a JSON response with the message key and the value Hello, World!.
class HelloWorldView(APIView):
    def get(self, request):
        return Response({"message": "Hello, World!"})
    
    def post(self, request):
        print('entered POST API call')
        # data = request.data
        file = request.FILES.get('img1')
        file2 = request.FILES.get('img2')

        img1 = pickle.loads(file.read())
        img2 = pickle.loads(file2.read())
        

        # img1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
        # img2 = cv2.imdecode(img2, cv2.IMREAD_COLOR)

        


        face_detector = MTCNN()
        faces =  face_detector.detect_faces(img1)
        for face in faces:
            x, y, width, height = face['box']
            x2, y2 = x + width, y + height

        #     # let me add extra buffer 10% of height and width to the cropped frame
            buffer_width = 10*img1.shape[1]//100
            buffer_height = 10*img1.shape[0]//100
            x, y, width, height = face['box']
            x2, y2 = x + width, y + height

            img = img1[max(0, y-buffer_height):min(img1.shape[0], y2+buffer_height), max(0, x-buffer_width):min(img1.shape[1], x2+buffer_width)]

        result = DeepFace.verify(img, img2, model_name='VGG-Face', distance_metric='euclidean', detector_backend='skip')
        
        
        return Response({"message": "Got some data!", "data": result,}, status=status.HTTP_200_OK)
