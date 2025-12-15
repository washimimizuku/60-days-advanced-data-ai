"""
Day 9: Data Lineage Tracking
Exercise: Build comprehensive data lineage tracking system

This exercise covers:
1. SQL parsing for automated lineage extraction
2. Multi-hop lineage tracking system
3. Column-level lineage with transformations
4. Impact analysis for change management
5. GDPR compliance lineage tracker
6. Data quality root cause analyzer
"""

import re
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

@dataclass
class LineageNode:
    """Represents a node in the lineage graph"""
    id: str
    type: str  # 'table', 'column', 'transformation'
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class LineageEdge:
    """Represents an edge in the lineage graph"""
    source: str
    target: str
    transformation_type: str
    transformation_logic: str
    created_at: datetime

class SQLLineageExtractor:
    """
    Exercise 1: SQL parsing for automated lineage extraction
    
    TODO: Implement SQL parsing to extract table and column lineage
    """
    
    def __init__(self):
        self.table_patterns = {
            'select_from': r'FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            'join': r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            'insert_into': r'INSERT\s+INTO\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            'update': r'UPDATE\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            'delete_from': r'DELETE\s+FROM\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        }
    
    def extract_table_lineage(self, sql_query: str) -> Dict[str, List[str]]:
        """
        TODO: Extract table-level lineage from SQL query
        
        Args:
            sql_query: SQL query string
            
        Returns:
            Dictionary with 'sources' and 'targets' lists
        """
        # TODO: Implement table lineage extraction
        # Hint: Use regex patterns to find source and target tables
        # Consider different SQL operations (SELECT, INSERT, UPDATE, DELETE)
        
        lineage = {
            'sources': [],
            'targets': []
        }
        
        # TODO: Extract source tables (FROM, JOIN clauses)
        # Example: FROM users u JOIN orders o -> sources: ['users', 'orders']
        sql_upper = sql_query.upper()
        
        # Find FROM clauses
        from_matches = re.findall(self.table_patterns['select_from'], sql_upper)
        for match in from_matches:
            table_name = match.split()[0] if ' ' in match else match
            if table_name.lower() not in lineage['sources']:
                lineage['sources'].append(table_name.lower())
        
        # Find JOIN clauses
        join_matches = re.findall(self.table_patterns['join'], sql_upper)
        for match in join_matches:
            table_name = match.split()[0] if ' ' in match else match
            if table_name.lower() not in lineage['sources']:
                lineage['sources'].append(table_name.lower())
        
        # TODO: Extract target tables (INSERT INTO, UPDATE clauses)
        # Example: INSERT INTO user_stats -> targets: ['user_stats']
        
        # Find INSERT INTO clauses
        insert_matches = re.findall(self.table_patterns['insert_into'], sql_upper)
        for match in insert_matches:
            table_name = match.split()[0] if ' ' in match else match
            if table_name.lower() not in lineage['targets']:
                lineage['targets'].append(table_name.lower())
        
        # Find UPDATE clauses
        update_matches = re.findall(self.table_patterns['update'], sql_upper)
        for match in update_matches:
            table_name = match.split()[0] if ' ' in match else match
            if table_name.lower() not in lineage['targets']:
                lineage['targets'].append(table_name.lower())
        
        return lineage
    
    def extract_column_lineage(self, sql_query: str) -> List[Dict[str, str]]:
        """
        TODO: Extract column-level lineage from SQL query
        
        Args:
            sql_query: SQL query string
            
        Returns:
            List of column mappings with source and target columns
        """
        # TODO: Implement column lineage extraction
        # This is more complex - need to parse SELECT clause and map to target columns
        # Consider transformations like UPPER(email), COUNT(*), SUM(amount)
        
        column_mappings = []
        
        # TODO: Parse SELECT clause to identify column transformations
        # Example: SELECT user_id, UPPER(email) as email_upper, COUNT(*) as cnt
        # Should return:
        # [
        #   {'source_table': 'users', 'source_column': 'user_id', 'target_column': 'user_id', 'transformation': 'direct'},
        #   {'source_table': 'users', 'source_column': 'email', 'target_column': 'email_upper', 'transformation': 'UPPER'},
        #   {'source_table': 'users', 'source_column': '*', 'target_column': 'cnt', 'transformation': 'COUNT'}
        # ]
        
        return column_mappings
    
    def parse_complex_query(self, sql_query: str) -> Dict[str, Any]:
        """
        TODO: Parse complex SQL with CTEs, subqueries, and window functions
        
        Args:
            sql_query: Complex SQL query
            
        Returns:
            Complete lineage information including intermediate steps
        """
        # TODO: Handle Common Table Expressions (WITH clauses)
        # TODO: Handle subqueries and nested SELECT statements
        # TODO: Handle window functions and partitioning
        
        return {
            'table_lineage': {},
            'column_lineage': [],
            'intermediate_steps': [],
            'complexity_score': 0
        }

class MultiHopLineageTracker:
    """
    Exercise 2: Multi-hop lineage tracking system
    
    TODO: Implement system to track lineage across multiple transformation steps
    """
    
    def __init__(self):
        self.nodes = {}  # node_id -> LineageNode
        self.edges = []  # List of LineageEdge
        self.adjacency_list = defaultdict(list)  # For efficient graph traversal
    
    def add_node(self, node_id: str, node_type: str, metadata: Dict[str, Any]) -> None:
        """
        TODO: Add a node to the lineage graph
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node ('table', 'column', 'transformation')
            metadata: Additional metadata about the node
        """
        # TODO: Create LineageNode and add to graph
        node = LineageNode(
            id=node_id,
            type=node_type,
            metadata=metadata,
            created_at=datetime.now()
        )
        self.nodes[node_id] = node
        
        # TODO: Update adjacency list for efficient traversal
        if node_id not in self.adjacency_list:
            self.adjacency_list[node_id] = []
    
    def add_edge(self, source: str, target: str, transformation_type: str, 
                 transformation_logic: str) -> None:
        """
        TODO: Add an edge to the lineage graph
        
        Args:
            source: Source node ID
            target: Target node ID
            transformation_type: Type of transformation
            transformation_logic: Logic/code for the transformation
        """
        # TODO: Create LineageEdge and add to graph
        edge = LineageEdge(
            source=source,
            target=target,
            transformation_type=transformation_type,
            transformation_logic=transformation_logic,
            created_at=datetime.now()
        )
        self.edges.append(edge)
        
        # TODO: Update adjacency list
        self.adjacency_list[source].append(target)
    
    def get_upstream_lineage(self, node_id: str, max_depth: int = 10) -> Set[str]:
        """
        TODO: Get all upstream dependencies for a node
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Set of upstream node IDs
        """
        # TODO: Implement BFS/DFS to find all upstream nodes
        # TODO: Respect max_depth to prevent infinite loops
        # TODO: Return set of node IDs that this node depends on
        
        upstream = set()
        # TODO: Implement traversal algorithm
        queue = deque([(node_id, 0)])
        visited = {node_id}
        
        while queue:
            current_node, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Find all edges where current_node is the target
            for edge in self.edges:
                if edge.target == current_node and edge.source not in visited:
                    upstream.add(edge.source)
                    visited.add(edge.source)
                    queue.append((edge.source, depth + 1))
        
        return upstream
    
    def get_downstream_lineage(self, node_id: str, max_depth: int = 10) -> Set[str]:
        """
        TODO: Get all downstream dependencies for a node
        
        Args:
            node_id: Starting node ID
            max_depth: Maximum depth to traverse
            
        Returns:
            Set of downstream node IDs
        """
        # TODO: Implement BFS/DFS to find all downstream nodes
        # TODO: Similar to upstream but traverse in opposite direction
        
        downstream = set()
        # TODO: Implement traversal algorithm
        return downstream
    
    def find_shortest_path(self, source: str, target: str) -> List[str]:
        """
        TODO: Find shortest path between two nodes
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of node IDs representing the shortest path
        """
        # TODO: Implement BFS to find shortest path
        # TODO: Return empty list if no path exists
        
        path = []
        # TODO: Implement shortest path algorithm
        return path

class ColumnLevelLineageTracker:
    """
    Exercise 3: Column-level lineage with transformations
    
    TODO: Implement detailed column-level lineage tracking
    """
    
    def __init__(self):
        self.column_mappings = []
        self.transformation_catalog = {}
    
    def add_column_mapping(self, source_table: str, source_column: str,
                          target_table: str, target_column: str,
                          transformation_type: str, transformation_logic: str) -> None:
        """
        TODO: Add column-level lineage mapping
        
        Args:
            source_table: Source table name
            source_column: Source column name
            target_table: Target table name
            target_column: Target column name
            transformation_type: Type of transformation
            transformation_logic: Actual transformation logic
        """
        # TODO: Create column mapping record
        # TODO: Store transformation details
        # TODO: Add to transformation catalog for reuse
        pass
    
    def trace_column_lineage(self, target_table: str, target_column: str) -> List[Dict[str, Any]]:
        """
        TODO: Trace lineage for a specific column back to its sources
        
        Args:
            target_table: Target table name
            target_column: Target column name
            
        Returns:
            List of lineage steps showing the full transformation chain
        """
        # TODO: Recursively trace column back to original sources
        # TODO: Include transformation details at each step
        # TODO: Handle complex transformations (aggregations, joins, etc.)
        
        lineage_chain = []
        # TODO: Implement recursive tracing
        return lineage_chain
    
    def find_pii_lineage(self) -> List[Dict[str, Any]]:
        """
        TODO: Find all columns that derive from PII data
        
        Returns:
            List of columns with PII lineage information
        """
        # TODO: Define PII patterns (email, phone, ssn, name, address)
        # TODO: Search through column mappings to find PII derivatives
        # TODO: Return detailed information about PII data flow
        
        pii_patterns = ['email', 'phone', 'ssn', 'name', 'address', 'dob']
        pii_lineage = []
        
        # TODO: Implement PII detection and lineage tracking
        return pii_lineage
    
    def analyze_transformation_complexity(self, target_table: str, target_column: str) -> Dict[str, Any]:
        """
        TODO: Analyze complexity of transformations for a column
        
        Args:
            target_table: Target table name
            target_column: Target column name
            
        Returns:
            Analysis of transformation complexity
        """
        # TODO: Count number of transformation steps
        # TODO: Identify types of transformations used
        # TODO: Calculate complexity score
        # TODO: Identify potential optimization opportunities
        
        return {
            'transformation_steps': 0,
            'transformation_types': [],
            'complexity_score': 0,
            'optimization_suggestions': []
        }

class ImpactAnalyzer:
    """
    Exercise 4: Impact analysis for change management
    
    TODO: Implement comprehensive impact analysis system
    """
    
    def __init__(self, lineage_tracker: MultiHopLineageTracker):
        self.lineage = lineage_tracker
        self.impact_cache = {}
    
    def analyze_change_impact(self, changed_node: str, change_type: str) -> Dict[str, Any]:
        """
        TODO: Analyze impact of changes to a data asset
        
        Args:
            changed_node: Node ID that will be changed
            change_type: Type of change ('schema', 'data', 'logic', 'removal')
            
        Returns:
            Comprehensive impact analysis
        """
        # TODO: Get all downstream dependencies
        # TODO: Categorize impacts by severity
        # TODO: Identify critical systems that might break
        # TODO: Generate recommendations for change management
        
        impact_analysis = {
            'changed_node': changed_node,
            'change_type': change_type,
            'downstream_impacts': [],
            'severity_breakdown': {'high': 0, 'medium': 0, 'low': 0},
            'affected_systems': [],
            'recommendations': [],
            'estimated_effort_hours': 0
        }
        
        # TODO: Implement impact analysis logic
        return impact_analysis
    
    def generate_change_plan(self, changes: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        TODO: Generate comprehensive change management plan
        
        Args:
            changes: List of planned changes
            
        Returns:
            Detailed change management plan
        """
        # TODO: Analyze all changes together
        # TODO: Identify dependencies between changes
        # TODO: Create optimal change order
        # TODO: Generate testing plan
        # TODO: Create rollback plan
        
        change_plan = {
            'changes': changes,
            'execution_order': [],
            'testing_plan': [],
            'rollback_plan': [],
            'risk_assessment': {},
            'timeline_estimate': {}
        }
        
        # TODO: Implement change planning logic
        return change_plan
    
    def simulate_outage_impact(self, failed_node: str) -> Dict[str, Any]:
        """
        TODO: Simulate impact of system outage
        
        Args:
            failed_node: Node ID that has failed
            
        Returns:
            Outage impact analysis
        """
        # TODO: Find all systems that depend on failed node
        # TODO: Calculate business impact (users affected, revenue impact)
        # TODO: Identify alternative data sources
        # TODO: Generate recovery plan
        
        outage_impact = {
            'failed_node': failed_node,
            'affected_systems': [],
            'business_impact': {},
            'alternative_sources': [],
            'recovery_plan': [],
            'estimated_downtime': 0
        }
        
        # TODO: Implement outage simulation
        return outage_impact

class GDPRLineageTracker:
    """
    Exercise 5: GDPR compliance lineage tracker
    
    TODO: Implement GDPR-specific lineage tracking for compliance
    """
    
    def __init__(self, lineage_tracker: MultiHopLineageTracker):
        self.lineage = lineage_tracker
        self.pii_catalog = {}
        self.retention_policies = {}
    
    def register_pii_data(self, table: str, column: str, pii_type: str, 
                         retention_period: int) -> None:
        """
        TODO: Register PII data for GDPR tracking
        
        Args:
            table: Table containing PII
            column: Column containing PII
            pii_type: Type of PII (email, name, address, etc.)
            retention_period: Retention period in days
        """
        # TODO: Add to PII catalog
        # TODO: Set retention policy
        # TODO: Mark for lineage tracking
        pass
    
    def find_personal_data_usage(self, user_id: str) -> List[Dict[str, Any]]:
        """
        TODO: Find all systems that contain data for a specific user
        
        Args:
            user_id: User identifier
            
        Returns:
            List of systems containing user's personal data
        """
        # TODO: Start with known PII tables
        # TODO: Follow lineage to find all downstream systems
        # TODO: Check if systems likely contain user data
        # TODO: Include retention policy information
        
        personal_data_locations = []
        # TODO: Implement personal data discovery
        return personal_data_locations
    
    def generate_data_deletion_plan(self, user_id: str) -> Dict[str, Any]:
        """
        TODO: Generate plan for deleting all user data (Right to be Forgotten)
        
        Args:
            user_id: User identifier
            
        Returns:
            Comprehensive data deletion plan
        """
        # TODO: Find all personal data locations
        # TODO: Sort by dependency order (delete downstream first)
        # TODO: Generate deletion scripts
        # TODO: Create verification steps
        
        deletion_plan = {
            'user_id': user_id,
            'systems_to_update': [],
            'deletion_order': [],
            'deletion_scripts': [],
            'verification_steps': [],
            'estimated_completion_time': 0
        }
        
        # TODO: Implement deletion planning
        return deletion_plan
    
    def audit_data_retention(self) -> Dict[str, Any]:
        """
        TODO: Audit data retention compliance
        
        Returns:
            Data retention audit report
        """
        # TODO: Check all PII data against retention policies
        # TODO: Identify data that should be deleted
        # TODO: Generate compliance report
        
        audit_report = {
            'audit_date': datetime.now().isoformat(),
            'compliant_systems': [],
            'non_compliant_systems': [],
            'data_to_delete': [],
            'recommendations': []
        }
        
        # TODO: Implement retention audit
        return audit_report

class DataQualityLineageAnalyzer:
    """
    Exercise 6: Data quality root cause analyzer
    
    TODO: Use lineage for data quality root cause analysis
    """
    
    def __init__(self, lineage_tracker: MultiHopLineageTracker):
        self.lineage = lineage_tracker
        self.quality_metrics = {}
    
    def register_quality_issue(self, dataset: str, issue_type: str, 
                              issue_description: str, severity: str) -> None:
        """
        TODO: Register a data quality issue
        
        Args:
            dataset: Dataset with quality issue
            issue_type: Type of issue (completeness, accuracy, consistency)
            issue_description: Description of the issue
            severity: Severity level (low, medium, high, critical)
        """
        # TODO: Store quality issue information
        # TODO: Timestamp the issue
        # TODO: Prepare for root cause analysis
        pass
    
    def analyze_quality_root_cause(self, dataset: str, issue_type: str) -> Dict[str, Any]:
        """
        TODO: Analyze root cause of data quality issue using lineage
        
        Args:
            dataset: Dataset with quality issue
            issue_type: Type of quality issue
            
        Returns:
            Root cause analysis results
        """
        # TODO: Get upstream lineage
        # TODO: Check quality metrics for upstream systems
        # TODO: Identify potential root causes
        # TODO: Calculate confidence scores
        # TODO: Generate recommendations
        
        analysis = {
            'dataset': dataset,
            'issue_type': issue_type,
            'potential_root_causes': [],
            'affected_downstream_systems': [],
            'confidence_scores': {},
            'recommendations': []
        }
        
        # TODO: Implement root cause analysis
        return analysis
    
    def propagate_quality_impact(self, source_dataset: str, quality_score: float) -> Dict[str, Any]:
        """
        TODO: Propagate quality impact through lineage
        
        Args:
            source_dataset: Dataset with known quality score
            quality_score: Quality score (0.0 to 1.0)
            
        Returns:
            Quality impact propagation results
        """
        # TODO: Get downstream lineage
        # TODO: Calculate quality degradation through transformations
        # TODO: Identify systems with quality below threshold
        # TODO: Generate quality improvement plan
        
        propagation_results = {
            'source_dataset': source_dataset,
            'source_quality_score': quality_score,
            'downstream_quality_scores': {},
            'systems_below_threshold': [],
            'improvement_plan': []
        }
        
        # TODO: Implement quality propagation
        return propagation_results

class LineageExerciseRunner:
    """Main class to run all lineage exercises"""
    
    def __init__(self):
        self.sql_extractor = SQLLineageExtractor()
        self.lineage_tracker = MultiHopLineageTracker()
        self.column_tracker = ColumnLevelLineageTracker()
        self.impact_analyzer = ImpactAnalyzer(self.lineage_tracker)
        self.gdpr_tracker = GDPRLineageTracker(self.lineage_tracker)
        self.quality_analyzer = DataQualityLineageAnalyzer(self.lineage_tracker)
    
    def run_exercise_1_sql_parsing(self):
        """Run Exercise 1: SQL parsing for lineage extraction"""
        print("=== Exercise 1: SQL Parsing for Lineage Extraction ===")
        
        # Test SQL queries
        test_queries = [
            """
            INSERT INTO analytics.user_metrics
            SELECT 
                u.user_id,
                u.email,
                COUNT(o.order_id) as order_count,
                SUM(o.total_amount) as total_spent
            FROM ecommerce.users u
            LEFT JOIN ecommerce.orders o ON u.user_id = o.user_id
            WHERE u.status = 'active'
            GROUP BY u.user_id, u.email
            """,
            """
            UPDATE products 
            SET stock_quantity = stock_quantity - oi.quantity
            FROM order_items oi
            WHERE products.product_id = oi.product_id
            """,
            """
            WITH monthly_sales AS (
                SELECT 
                    DATE_TRUNC('month', created_at) as month,
                    SUM(total_amount) as monthly_revenue
                FROM orders
                WHERE status = 'completed'
                GROUP BY DATE_TRUNC('month', created_at)
            )
            SELECT * FROM monthly_sales
            WHERE monthly_revenue > 10000
            """
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nTest Query {i}:")
            print("Table Lineage:", self.sql_extractor.extract_table_lineage(query))
            print("Column Lineage:", self.sql_extractor.extract_column_lineage(query))
    
    def run_exercise_2_multi_hop_lineage(self):
        """Run Exercise 2: Multi-hop lineage tracking"""
        print("\n=== Exercise 2: Multi-hop Lineage Tracking ===")
        
        # TODO: Add sample nodes and edges
        # TODO: Test upstream and downstream lineage
        # TODO: Test shortest path finding
        
        print("Multi-hop lineage tracking exercise completed")
    
    def run_exercise_3_column_lineage(self):
        """Run Exercise 3: Column-level lineage tracking"""
        print("\n=== Exercise 3: Column-level Lineage Tracking ===")
        
        # TODO: Add sample column mappings
        # TODO: Test column lineage tracing
        # TODO: Test PII lineage detection
        
        print("Column-level lineage tracking exercise completed")
    
    def run_exercise_4_impact_analysis(self):
        """Run Exercise 4: Impact analysis"""
        print("\n=== Exercise 4: Impact Analysis ===")
        
        # TODO: Test change impact analysis
        # TODO: Test change plan generation
        # TODO: Test outage simulation
        
        print("Impact analysis exercise completed")
    
    def run_exercise_5_gdpr_compliance(self):
        """Run Exercise 5: GDPR compliance tracking"""
        print("\n=== Exercise 5: GDPR Compliance Tracking ===")
        
        # TODO: Register sample PII data
        # TODO: Test personal data discovery
        # TODO: Test deletion plan generation
        
        print("GDPR compliance tracking exercise completed")
    
    def run_exercise_6_quality_analysis(self):
        """Run Exercise 6: Data quality root cause analysis"""
        print("\n=== Exercise 6: Data Quality Root Cause Analysis ===")
        
        # TODO: Register sample quality issues
        # TODO: Test root cause analysis
        # TODO: Test quality impact propagation
        
        print("Data quality analysis exercise completed")
    
    def run_all_exercises(self):
        """Run all lineage tracking exercises"""
        print("🚀 Starting Data Lineage Tracking Exercises")
        print("=" * 60)
        
        self.run_exercise_1_sql_parsing()
        self.run_exercise_2_multi_hop_lineage()
        self.run_exercise_3_column_lineage()
        self.run_exercise_4_impact_analysis()
        self.run_exercise_5_gdpr_compliance()
        self.run_exercise_6_quality_analysis()
        
        print("\n✅ All exercises completed!")
        print("\nNext steps:")
        print("1. Complete the TODO items in each exercise")
        print("2. Test with real SQL queries and data")
        print("3. Integrate with actual data catalog systems")
        print("4. Add monitoring and alerting capabilities")
        print("5. Implement performance optimizations")

def main():
    """Run the data lineage tracking exercises"""
    runner = LineageExerciseRunner()
    runner.run_all_exercises()

if __name__ == "__main__":
    main()
