from rest_framework import serializers

class SatireDetectionSerializer(serializers.Serializer):
    text = serializers.CharField(required=True)
    
class SatireDetectionResultSerializer(serializers.Serializer):
    prediction = serializers.CharField()
    probability = serializers.FloatField()
  