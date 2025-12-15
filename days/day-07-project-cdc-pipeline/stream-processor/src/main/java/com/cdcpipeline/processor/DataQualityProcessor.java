package com.cdcpipeline.processor;

import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.kstream.Branched;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Validates data quality for all incoming events
 * Routes invalid events to dead letter queue and generates quality metrics
 */
public class DataQualityProcessor {
    
    private static final Logger logger = LoggerFactory.getLogger(DataQualityProcessor.class);
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    public void buildTopology(StreamsBuilder builder) {
        logger.info("Building data quality topology");
        
        // Process all event types
        processOrderEvents(builder);
        processUserEvents(builder);
        processProductEvents(builder);
        
        logger.info("Data quality topology built successfully");
    }
    
    private void processOrderEvents(StreamsBuilder builder) {
        KStream<String, String> orders = builder.stream("orders");
        
        Map<String, KStream<String, String>> branches = orders
            .split()
            .branch((key, value) -> isValidOrder(value), Branched.as("valid"))
            .defaultBranch(Branched.as("invalid"));
        
        // Send invalid orders to DLQ
        branches.get("invalid")
            .mapValues(value -> createDLQEvent("orders", value, validateOrder(value)))
            .to("dlq-events");
        
        // Generate quality metrics for orders
        generateQualityMetrics(branches.get("valid"), branches.get("invalid"), "orders");
    }
    
    private void processUserEvents(StreamsBuilder builder) {
        KStream<String, String> users = builder.stream("users");
        
        Map<String, KStream<String, String>> branches = users
            .split()
            .branch((key, value) -> isValidUser(value), Branched.as("valid"))
            .defaultBranch(Branched.as("invalid"));
        
        // Send invalid users to DLQ
        branches.get("invalid")
            .mapValues(value -> createDLQEvent("users", value, validateUser(value)))
            .to("dlq-events");
        
        // Generate quality metrics for users
        generateQualityMetrics(branches.get("valid"), branches.get("invalid"), "users");
    }
    
    private void processProductEvents(StreamsBuilder builder) {
        KStream<String, String> products = builder.stream("products");
        
        Map<String, KStream<String, String>> branches = products
            .split()
            .branch((key, value) -> isValidProduct(value), Branched.as("valid"))
            .defaultBranch(Branched.as("invalid"));
        
        // Send invalid products to DLQ
        branches.get("invalid")
            .mapValues(value -> createDLQEvent("products", value, validateProduct(value)))
            .to("dlq-events");
        
        // Generate quality metrics for products
        generateQualityMetrics(branches.get("valid"), branches.get("invalid"), "products");
    }
    
    private boolean isValidOrder(String orderJson) {
        return validateOrder(orderJson).isEmpty();
    }
    
    private boolean isValidUser(String userJson) {
        return validateUser(userJson).isEmpty();
    }
    
    private boolean isValidProduct(String productJson) {
        return validateProduct(productJson).isEmpty();
    }
    
    private List<String> validateOrder(String orderJson) {
        List<String> errors = new ArrayList<>();
        
        try {
            JsonNode orderNode = objectMapper.readTree(orderJson);
            
            // Required fields validation
            if (!orderNode.has("order_id") || orderNode.path("order_id").isNull()) {
                errors.add("Missing required field: order_id");
            }
            
            if (!orderNode.has("user_id") || orderNode.path("user_id").isNull()) {
                errors.add("Missing required field: user_id");
            }
            
            if (!orderNode.has("total_amount") || orderNode.path("total_amount").isNull()) {
                errors.add("Missing required field: total_amount");
            } else {
                double totalAmount = orderNode.path("total_amount").asDouble();
                if (totalAmount <= 0) {
                    errors.add("Total amount must be positive");
                }
            }
            
            if (!orderNode.has("status") || orderNode.path("status").isNull()) {
                errors.add("Missing required field: status");
            } else {
                String status = orderNode.path("status").asText();
                if (!isValidOrderStatus(status)) {
                    errors.add("Invalid order status: " + status);
                }
            }
            
        } catch (Exception e) {
            errors.add("Invalid JSON format: " + e.getMessage());
        }
        
        return errors;
    }
    
    private List<String> validateUser(String userJson) {
        List<String> errors = new ArrayList<>();
        
        try {
            JsonNode userNode = objectMapper.readTree(userJson);
            
            // Required fields validation
            if (!userNode.has("user_id") || userNode.path("user_id").isNull()) {
                errors.add("Missing required field: user_id");
            }
            
            if (!userNode.has("email") || userNode.path("email").isNull()) {
                errors.add("Missing required field: email");
            } else {
                String email = userNode.path("email").asText();
                if (!isValidEmail(email)) {
                    errors.add("Invalid email format: " + email);
                }
            }
            
            if (!userNode.has("first_name") || userNode.path("first_name").isNull()) {
                errors.add("Missing required field: first_name");
            }
            
            if (!userNode.has("last_name") || userNode.path("last_name").isNull()) {
                errors.add("Missing required field: last_name");
            }
            
        } catch (Exception e) {
            errors.add("Invalid JSON format: " + e.getMessage());
        }
        
        return errors;
    }
    
    private List<String> validateProduct(String productJson) {
        List<String> errors = new ArrayList<>();
        
        try {
            JsonNode productNode = objectMapper.readTree(productJson);
            
            // Required fields validation
            if (!productNode.has("product_id") || productNode.path("product_id").isNull()) {
                errors.add("Missing required field: product_id");
            }
            
            if (!productNode.has("name") || productNode.path("name").isNull()) {
                errors.add("Missing required field: name");
            }
            
            if (!productNode.has("price") || productNode.path("price").isNull()) {
                errors.add("Missing required field: price");
            } else {
                double price = productNode.path("price").asDouble();
                if (price <= 0) {
                    errors.add("Price must be positive");
                }
            }
            
            if (!productNode.has("stock_quantity") || productNode.path("stock_quantity").isNull()) {
                errors.add("Missing required field: stock_quantity");
            } else {
                int stockQuantity = productNode.path("stock_quantity").asInt();
                if (stockQuantity < 0) {
                    errors.add("Stock quantity cannot be negative");
                }
            }
            
        } catch (Exception e) {
            errors.add("Invalid JSON format: " + e.getMessage());
        }
        
        return errors;
    }
    
    private boolean isValidOrderStatus(String status) {
        return status != null && 
               (status.equals("pending") || status.equals("processing") || 
                status.equals("completed") || status.equals("cancelled"));
    }
    
    private boolean isValidEmail(String email) {
        return email != null && email.contains("@") && email.contains(".");
    }
    
    private String createDLQEvent(String topic, String originalEvent, List<String> errors) {
        try {
            return String.format(
                "{\"source_topic\":\"%s\",\"original_event\":%s,\"errors\":%s,\"timestamp\":\"%s\"}",
                topic,
                originalEvent,
                objectMapper.writeValueAsString(errors),
                java.time.Instant.now().toString()
            );
        } catch (Exception e) {
            logger.error("Failed to create DLQ event: {}", e.getMessage());
            return String.format(
                "{\"source_topic\":\"%s\",\"error\":\"Failed to process event\",\"timestamp\":\"%s\"}",
                topic,
                java.time.Instant.now().toString()
            );
        }
    }
    
    private void generateQualityMetrics(KStream<String, String> validEvents, 
                                      KStream<String, String> invalidEvents, 
                                      String eventType) {
        // Count valid events
        validEvents
            .mapValues(value -> String.format(
                "{\"metric_name\":\"%s_valid_events\",\"metric_value\":1,\"status\":\"ok\",\"measured_at\":\"%s\"}",
                eventType,
                java.time.Instant.now().toString()
            ))
            .to("data-quality-events");
        
        // Count invalid events
        invalidEvents
            .mapValues(value -> String.format(
                "{\"metric_name\":\"%s_invalid_events\",\"metric_value\":1,\"status\":\"warning\",\"measured_at\":\"%s\"}",
                eventType,
                java.time.Instant.now().toString()
            ))
            .to("data-quality-events");
    }
}