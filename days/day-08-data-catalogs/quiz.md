# Day 8 Quiz: Data Catalogs - DataHub, Amundsen

## Questions

### 1. What is the primary purpose of a data catalog in modern data architecture?
- a) To store actual data files
- b) To provide centralized metadata management and data discovery
- c) To replace databases
- d) To perform data transformations

### 2. Which architecture component is unique to DataHub compared to Amundsen?
- a) Neo4j graph database
- b) Elasticsearch for search
- c) Kafka for real-time metadata updates
- d) Flask web framework

### 3. What type of lineage tracking does DataHub support that Amundsen typically doesn't?
- a) Table-level lineage
- b) Column-level lineage
- c) Schema-level lineage
- d) Database-level lineage

### 4. In DataHub's architecture, what is the role of the GMS (Generalized Metadata Service)?
- a) User interface rendering
- b) Search functionality
- c) Backend API for metadata operations
- d) Data storage

### 5. Which database does Amundsen use for storing relationship information?
- a) MySQL
- b) PostgreSQL
- c) Neo4j
- d) MongoDB

### 6. What is a MetadataChangeProposal (MCP) in DataHub?
- a) A user interface component
- b) A way to propose changes to metadata
- c) A search query format
- d) A lineage visualization

### 7. Which governance capability is typically easier to implement in a data catalog?
- a) Data encryption
- b) Access control enforcement
- c) Metadata tagging and classification
- d) Data backup

### 8. What is the main advantage of using Kafka in DataHub's architecture?
- a) Better search performance
- b) Real-time metadata updates
- c) Improved user interface
- d) Lower storage costs

### 9. In data catalog terminology, what does "data lineage" refer to?
- a) The age of the data
- b) The size of the dataset
- c) The flow and transformation of data through systems
- d) The number of users accessing the data

### 10. Which approach is recommended for ensuring high adoption of a data catalog?
- a) Making it mandatory for all users
- b) Focusing only on technical metadata
- c) Providing clear business value and easy discovery
- d) Restricting access to data stewards only

---

## Answers

### 1. What is the primary purpose of a data catalog in modern data architecture?
**Answer: b) To provide centralized metadata management and data discovery**

**Explanation:** Data catalogs serve as "Google for your data" - they provide centralized metadata management, data discovery, lineage tracking, and governance capabilities. They don't store actual data (that's what databases and data lakes do), replace databases, or perform transformations. Their core value is helping organizations find, understand, and govern their data assets through comprehensive metadata management.

---

### 2. Which architecture component is unique to DataHub compared to Amundsen?
**Answer: c) Kafka for real-time metadata updates**

**Explanation:** DataHub uses Kafka for real-time metadata updates, allowing immediate propagation of metadata changes across the system. Amundsen typically uses batch-based updates. Both systems use Elasticsearch for search, but Amundsen uses Neo4j (not DataHub), and neither uses Flask as their primary framework (Amundsen uses Flask for frontend, DataHub uses React).

---

### 3. What type of lineage tracking does DataHub support that Amundsen typically doesn't?
**Answer: b) Column-level lineage**

**Explanation:** DataHub supports fine-grained column-level lineage tracking, showing how individual columns flow and transform through different datasets. Amundsen typically provides table-level lineage. This column-level detail is crucial for impact analysis, compliance, and understanding data transformations at a granular level.

---

### 4. In DataHub's architecture, what is the role of the GMS (Generalized Metadata Service)?
**Answer: c) Backend API for metadata operations**

**Explanation:** The GMS (Generalized Metadata Service) is DataHub's backend service that handles all metadata operations including ingestion, storage, retrieval, and serving metadata through APIs. It's the core backend component that the frontend and other services interact with. The frontend handles UI rendering, Elasticsearch provides search, and various databases handle storage.

---

### 5. Which database does Amundsen use for storing relationship information?
**Answer: c) Neo4j**

**Explanation:** Amundsen uses Neo4j, a graph database, to store and query relationships between data assets. This graph structure is particularly well-suited for representing complex relationships between tables, columns, users, and other metadata entities. The graph model enables powerful relationship queries and traversals that are central to Amundsen's functionality.

---

### 6. What is a MetadataChangeProposal (MCP) in DataHub?
**Answer: b) A way to propose changes to metadata**

**Explanation:** A MetadataChangeProposal (MCP) is DataHub's mechanism for proposing changes to metadata. It's a structured way to submit metadata updates, whether from ingestion pipelines, manual updates, or automated processes. MCPs go through DataHub's processing pipeline and can trigger real-time updates via Kafka. It's not a UI component, search format, or visualization tool.

---

### 7. Which governance capability is typically easier to implement in a data catalog?
**Answer: c) Metadata tagging and classification**

**Explanation:** Metadata tagging and classification are core features of data catalogs and are relatively straightforward to implement. Data catalogs excel at organizing and categorizing data assets through tags, glossary terms, and classifications. Data encryption, access control enforcement, and backup are typically handled by the underlying data systems, not the catalog itself.

---

### 8. What is the main advantage of using Kafka in DataHub's architecture?
**Answer: b) Real-time metadata updates**

**Explanation:** Kafka enables real-time metadata updates in DataHub, allowing immediate propagation of changes across the system. When metadata is updated, Kafka streams these changes to all relevant components, ensuring the catalog stays current. This is a significant advantage over batch-based systems where metadata updates might be delayed by hours or days.

---

### 9. In data catalog terminology, what does "data lineage" refer to?
**Answer: c) The flow and transformation of data through systems**

**Explanation:** Data lineage tracks the flow and transformation of data from source to destination, showing how data moves through different systems, what transformations are applied, and where it ends up. This is crucial for impact analysis (understanding what breaks if you change something), compliance (tracking data origins), and debugging data quality issues. It's not about data age, size, or user access patterns.

---

### 10. Which approach is recommended for ensuring high adoption of a data catalog?
**Answer: c) Providing clear business value and easy discovery**

**Explanation:** High adoption of data catalogs comes from demonstrating clear business value and making data discovery easy and intuitive. Users need to see immediate benefits like finding data faster, understanding data better, and avoiding duplicate work. Making it mandatory without value leads to resistance, focusing only on technical metadata misses business context, and restricting access defeats the purpose of democratizing data discovery.

---

## Score Interpretation

- **9-10 correct**: Excellent! You have a strong understanding of data catalog concepts and architecture
- **7-8 correct**: Good job! Review the questions you missed and focus on DataHub vs Amundsen differences
- **5-6 correct**: You're on the right track. Study the architecture components and governance concepts more
- **Below 5**: Review the theory section and practice with the hands-on exercises

---

## Key Concepts to Remember

1. **Data catalogs provide metadata management and discovery**, not data storage
2. **DataHub uses Kafka for real-time updates**, Amundsen uses batch processing
3. **DataHub supports column-level lineage**, Amundsen typically table-level
4. **GMS is DataHub's backend API service** for metadata operations
5. **Amundsen uses Neo4j graph database** for relationship storage
6. **MCPs are DataHub's metadata change mechanism** for updates
7. **Metadata tagging is easier to implement** than data-level security
8. **Kafka enables real-time metadata propagation** in DataHub
9. **Data lineage tracks data flow and transformations** through systems
10. **User adoption requires clear business value** and easy discovery

---

## Data Catalog Best Practices

### Implementation Strategy
- **Start with high-value datasets** that are frequently used
- **Focus on business metadata** alongside technical metadata
- **Implement automated metadata extraction** to reduce manual effort
- **Establish clear ownership and governance** processes

### Metadata Quality
- **Ensure comprehensive descriptions** for all datasets
- **Implement consistent tagging strategies** across the organization
- **Maintain up-to-date lineage information** through automation
- **Regular quality assessments** and cleanup processes

### User Adoption
- **Provide training and documentation** for end users
- **Demonstrate clear ROI** through time savings and better decisions
- **Make search intuitive and fast** with good ranking algorithms
- **Integrate with existing workflows** and tools

### Governance Integration
- **Align with data governance policies** and procedures
- **Implement automated policy enforcement** where possible
- **Track compliance metrics** and generate reports
- **Establish clear escalation processes** for violations

### Technical Considerations
- **Plan for scale** with appropriate infrastructure sizing
- **Implement proper monitoring** and alerting
- **Ensure high availability** for critical business processes
- **Regular backup and disaster recovery** procedures

Ready to move on to Day 9! 🚀
