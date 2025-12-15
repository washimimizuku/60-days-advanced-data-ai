package com.cdcpipeline;

import com.cdcpipeline.processor.RevenueAnalyticsProcessor;
import com.cdcpipeline.processor.InventoryMonitoringProcessor;
import com.cdcpipeline.processor.DataQualityProcessor;
import com.cdcpipeline.config.StreamsConfig;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.Topology;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CountDownLatch;

/**
 * Main application class for the CDC Pipeline Stream Processor
 * 
 * This application processes real-time events from the CDC pipeline:
 * - Revenue analytics with windowed aggregations
 * - Inventory monitoring and alerting
 * - Data quality validation and error handling
 */
public class StreamProcessorApplication {
    
    private static final Logger logger = LoggerFactory.getLogger(StreamProcessorApplication.class);
    
    public static void main(String[] args) {
        logger.info("Starting CDC Pipeline Stream Processor...");
        
        try {
            // Build the topology
            StreamsBuilder builder = new StreamsBuilder();
            
            // Initialize processors
            RevenueAnalyticsProcessor revenueProcessor = new RevenueAnalyticsProcessor();
            InventoryMonitoringProcessor inventoryProcessor = new InventoryMonitoringProcessor();
            DataQualityProcessor dataQualityProcessor = new DataQualityProcessor();
            
            // Build processing topology
            revenueProcessor.buildTopology(builder);
            inventoryProcessor.buildTopology(builder);
            dataQualityProcessor.buildTopology(builder);
            
            Topology topology = builder.build();
            logger.info("Stream processing topology:\n{}", topology.describe());
            
            // Create and start Kafka Streams
            KafkaStreams streams = new KafkaStreams(topology, StreamsConfig.getProperties());
            
            // Add shutdown hook
            CountDownLatch latch = new CountDownLatch(1);
            Runtime.getRuntime().addShutdownHook(new Thread("streams-shutdown-hook") {
                @Override
                public void run() {
                    logger.info("Shutting down stream processor...");
                    streams.close();
                    latch.countDown();
                }
            });
            
            // Start processing
            streams.start();
            logger.info("Stream processor started successfully");
            
            // Wait for shutdown
            latch.await();
            
        } catch (Exception e) {
            logger.error("Failed to start stream processor", e);
            System.exit(1);
        }
        
        logger.info("Stream processor shutdown complete");
    }
}