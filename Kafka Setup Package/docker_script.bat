docker-compose -f docker-compose.yml up -d
timeout /t 5
docker exec kafka /bin/bash -c "cd /opt/kafka/bin; kafka-topics.sh --create --topic ml-raw-dns --bootstrap-server localhost:9092"
docker exec kafka /bin/bash -c "cd /opt/kafka/bin; kafka-topics.sh --create --topic ml-dns-predictions --bootstrap-server localhost:9092"
pip install -r requirements.txt
python data_ingestion_script.py

