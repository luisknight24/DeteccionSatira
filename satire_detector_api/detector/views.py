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
            prediction, prob_satira, features_dict = processor.predict_satire(text)
            result_data = {
                "prediction": prediction,
                "probability": prob_satira,
                "metrics": {
                    "irony_score": limpiar_valor_float(features_dict.get("irony_score", 0)),
                    "LexicalDiversity": limpiar_valor_float(features_dict.get("LexicalDiversity", 0)),
                    "Flesch Score": limpiar_valor_float(features_dict.get("Flesch Score", 0)),
                    "Unusual Word Frequency": limpiar_valor_float(features_dict.get("Unusual Word Frequency", 0)),
                    "prop_NOUN": limpiar_valor_float(features_dict.get("prop_NOUN", 0)),
                    "prop_VERB": limpiar_valor_float(features_dict.get("prop_VERB", 0)),
                    "prop_ADJ": limpiar_valor_float(features_dict.get("prop_ADJ", 0))
                }
            }
            result_serializer = SatireDetectionResultSerializer(result_data)
            return Response(result_serializer.data, status=status.HTTP_200_OK)
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        