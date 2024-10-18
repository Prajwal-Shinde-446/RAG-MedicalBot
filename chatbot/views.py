from django.shortcuts import render
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
import os
from .utils import create_VDB, retrieve_answers

# Create your views here.
@api_view(['POST'])
def upload_doc(request):
    # Check if file is in the request
    if 'file' not in request.FILES:
        return Response({"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST)
    
    file = request.FILES['file']

    # Validate if it's a CSV
    if not file.name.endswith('.csv'):
        return Response({"error": "Invalid file type. Please upload a CSV file."}, status=status.HTTP_400_BAD_REQUEST)

    # Create file path
    file_path = os.path.join(settings.STATIC_ROOT, 'files', file.name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the file to static/files
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    # Convert CSV to text
    try:
        index = create_VDB(file_url=file_path)
    except Exception as e:
                return Response({'error': 'Could not create pinecone index'}, status=status.HTTP_400_BAD_REQUEST)

    # Return the converted text
    return Response({
                'message': 'Document uploaded and indexed successfully'
            }, status=status.HTTP_200_OK)



@api_view(['GET'])
# @permission_classes([IsAuthenticated])
def chatbot_response(request):
    if request.method == 'GET':
        query = request.data.get('query',None)
        document_id = request.data.get('document_id',None)

        index_name ='moodle'

        if not query:
            return Response({'message': 'how can i assist you?'})
        
        if document_id:
            try:
                response = retrieve_answers(query=query, document_id=document_id)
            except Exception as e:
                print(e)
                return Response({'error': 'Could not create response'}, status=status.HTTP_400_BAD_REQUEST)

        else:
            try:
                response = retrieve_answers(query=query)
            except Exception as e:
                print(e)
                return Response({'error': 'Could not create response'}, status=status.HTTP_400_BAD_REQUEST)
            
        return Response({'response': response}, status=status.HTTP_200_OK)


             
            