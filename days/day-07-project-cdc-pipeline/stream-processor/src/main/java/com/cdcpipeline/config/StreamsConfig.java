package com.cdcpipeline.config;

import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.StreamsConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Configuration class for Kafka Streams application
 * Provides centralized configuration management with environment variable support
 */
public class StreamsConfig {
    
    private static final Logger logger = LoggerFactory.getLogger(StreamsConfig.class);
    
    public static Properties getProperties() {
        Properties props = new Properties();
        
        // Application configuration
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, 
                  getEnvOrDefault("APPLICATION_ID", "cdc-stream-processor"));
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, 
                  getEnvOrDefault("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"));
        
        // Serialization
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        
        // Schema Registry
        String schemaRegistryUrl = getEnvOrDefault("SCHEMA_REGISTRY_URL", "http://localhost:8081");
        props.put("schema.registry.url", schemaRegistryUrl);
        
        // Performance tuning
        props.put(StreamsConfig.NUM_STREAM_THREADS_CONFIG, 
                  Integer.parseInt(getEnvOrDefault("STREAM_THREADS", "2")));
        props.put(StreamsConfig.COMMIT_INTERVAL_MS_CONFIG, 
                  Integer.parseInt(getEnvOrDefault("COMMIT_INTERVAL_MS", "1000")));
        props.put(StreamsConfig.CACHE_MAX_BYTES_BUFFERING_CONFIG, 
                  Long.parseLong(getEnvOrDefault("CACHE_MAX_BYTES", "10485760"))); // 10MB
        
        // Consumer configuration
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        props.put(ConsumerConfig.SESSION_TIMEOUT_MS_CONFIG, 30000);
        props.put(ConsumerConfig.HEARTBEAT_INTERVAL_MS_CONFIG, 10000);
        
        // Producer configuration
        props.put(ProducerConfig.ACKS_CONFIG, "all");
        props.put(ProducerConfig.RETRIES_CONFIG, Integer.MAX_VALUE);
        props.put(ProducerConfig.MAX_IN_FLIGHT_REQUESTS_PER_CONNECTION, 1);
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        
        // Error handling
        props.put(StreamsConfig.DEFAULT_DESERIALIZATION_EXCEPTION_HANDLER_CLASS_CONFIG,
                  "org.apache.kafka.streams.errors.LogAndContinueExceptionHandler");
        props.put(StreamsConfig.DEFAULT_PRODUCTION_EXCEPTION_HANDLER_CLASS_CONFIG,
                  "org.apache.kafka.streams.errors.DefaultProductionExceptionHandler");
        
        // Monitoring
        props.put(StreamsConfig.METRICS_RECORDING_LEVEL_CONFIG, "INFO");
        props.put(StreamsConfig.METRICS_SAMPLE_WINDOW_MS_CONFIG, 30000);
        
        logger.info("Kafka Streams configuration loaded with bootstrap servers: {}", 
                   props.getProperty(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG));
        
        return props;
    }
    
    private static String getEnvOrDefault(String envVar, String defaultValue) {
        String value = System.getenv(envVar);
        return value != null ? value : defaultValue;
    }
}