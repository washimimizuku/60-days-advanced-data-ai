package com.cdcpipeline.processor;

import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.kstream.KStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Monitors product inventory levels and generates alerts
 * Processes product update events to detect low stock conditions
 */
public class InventoryMonitoringProcessor {
    
    private static final Logger logger = LoggerFactory.getLogger(InventoryMonitoringProcessor.class);
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    public void buildTopology(StreamsBuilder builder) {
        logger.info("Building inventory monitoring topology");
        
        // Process product events
        KStream<String, String> products = builder.stream("products");
        
        // Monitor inventory levels and generate alerts
        KStream<String, String> inventoryAlerts = products
            .filter((key, value) -> {
                try {
                    JsonNode productNode = objectMapper.readTree(value);
                    int stockQuantity = productNode.path("stock_quantity").asInt();
                    int minThreshold = productNode.path("min_threshold").asInt(10);
                    
                    // Generate alert if stock is below threshold
                    return stockQuantity <= minThreshold;
                } catch (Exception e) {
                    logger.warn("Failed to parse product event: {}", e.getMessage());
                    return false;
                }
            })
            .mapValues(value -> {
                try {
                    JsonNode productNode = objectMapper.readTree(value);
                    int productId = productNode.path("product_id").asInt();
                    String productName = productNode.path("name").asText();
                    int stockQuantity = productNode.path("stock_quantity").asInt();
                    int minThreshold = productNode.path("min_threshold").asInt(10);
                    
                    String alertLevel;
                    if (stockQuantity == 0) {
                        alertLevel = "out_of_stock";
                    } else if (stockQuantity <= minThreshold / 2) {
                        alertLevel = "critical";
                    } else {
                        alertLevel = "low";
                    }
                    
                    return String.format(
                        "{\"product_id\":%d,\"product_name\":\"%s\",\"current_stock\":%d,\"threshold\":%d,\"alert_level\":\"%s\",\"created_at\":\"%s\"}",
                        productId,
                        productName,
                        stockQuantity,
                        minThreshold,
                        alertLevel,
                        java.time.Instant.now().toString()
                    );
                } catch (Exception e) {
                    logger.error("Failed to create inventory alert: {}", e.getMessage());
                    return null;
                }
            })
            .filter((key, value) -> value != null);
        
        // Send alerts to inventory-alerts topic
        inventoryAlerts.to("inventory-alerts");
        
        logger.info("Inventory monitoring topology built successfully");
    }
}