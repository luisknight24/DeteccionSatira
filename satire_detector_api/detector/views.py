from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import math
#from  detector. import SatireDetectionSerializer, SatireDetectionResultSerializer
#from detector.utils.text_processor import TextProcessor
from detector.serializers import SatireDetectionSerializer, SatireDetectionResultSerializer
from detector.utils.text_processor import TextProcessor  
def limpiar_valor_float(valor):
    """Reemplaza NaN o infinitos por un valor numérico válido"""
    if isinstance(valor, float):
        if math.isnan(valor) or math.isinf(valor):
            return 0.0
    return valor

class SatireDetectionAPI(APIView):

    def post(self, request):
        serializer = SatireDetectionSerializer(data=request.data)
        if serializer.is_valid():
            text = serializer.validated_data['text']
            processor = TextProcessor()
            prediction, prob_satira = processor.predict_satire(text)
            result_data = {
            
            "prediction": prediction,
            "probability": prob_satira
            }
            result_serializer = SatireDetectionResultSerializer(result_data)
            return Response(result_serializer.data, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        