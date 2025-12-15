package com.cdcpipeline.processor;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.kstream.*;
import org.apache.kafka.streams.kstream.Materialized;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.time.Duration;

/**
 * Processes order events to generate real-time revenue analytics
 * Calculates windowed aggregations for revenue, order counts, and average order values
 */
public class RevenueAnalyticsProcessor {
    
    private static final Logger logger = LoggerFactory.getLogger(RevenueAnalyticsProcessor.class);
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    public void buildTopology(StreamsBuilder builder) {
        logger.info("Building revenue analytics topology");
        
        // Process order events
        KStream<String, String> orders = builder.stream("orders");
        
        // Filter completed orders and extract revenue data
        KStream<String, Double> completedOrders = orders
            .filter((key, value) -> {
                try {
                    JsonNode orderNode = objectMapper.readTree(value);
                    String status = orderNode.path("status").asText();
                    return "completed".equals(status);
                } catch (Exception e) {
                    logger.warn("Failed to parse order event: {}", e.getMessage());
                    return false;
                }
            })
            .mapValues(value -> {
                try {
                    JsonNode orderNode = objectMapper.readTree(value);
                    return orderNode.path("total_amount").asDouble();
                } catch (Exception e) {
                    logger.warn("Failed to extract total_amount: {}", e.getMessage());
                    return 0.0;
                }
            });
        
        // 5-minute windowed revenue aggregation
        KTable<Windowed<String>, Double> revenueByWindow = completedOrders
            .groupBy((key, value) -> "revenue")
            .windowedBy(TimeWindows.of(Duration.ofMinutes(5)).advanceBy(Duration.ofMinutes(1)))
            .aggregate(
                () -> 0.0,
                (key, value, aggregate) -> aggregate + value,
                Materialized.with(Serdes.String(), Serdes.Double())
            );
        
        // Count orders in 5-minute windows
        KTable<Windowed<String>, Long> orderCountByWindow = completedOrders
            .groupBy((key, value) -> "orders")
            .windowedBy(TimeWindows.of(Duration.ofMinutes(5)).advanceBy(Duration.ofMinutes(1)))
            .count(Materialized.with(Serdes.String(), Serdes.Long()));
        
        // Calculate average order value
        KStream<String, String> revenueAnalytics = revenueByWindow
            .toStream()
            .join(orderCountByWindow.toStream(),
                (revenue, count) -> {
                    double avgOrderValue = count > 0 ? revenue / count : 0.0;
                    return String.format(
                        "{\"window_start\":\"%s\",\"window_end\":\"%s\",\"total_revenue\":%.2f,\"order_count\":%d,\"avg_order_value\":%.2f,\"processed_at\":\"%s\"}",
                        "2023-01-01T00:00:00Z", // Placeholder - should use actual window times
                        "2023-01-01T00:05:00Z",
                        revenue,
                        count,
                        avgOrderValue,
                        java.time.Instant.now().toString()
                    );
                },
                JoinWindows.of(Duration.ofMinutes(1))
            )
            .selectKey((windowedKey, value) -> "revenue-analytics");
        
        // Send to analytics topic
        revenueAnalytics.to("revenue-analytics");
        
        logger.info("Revenue analytics topology built successfully");
    }
}