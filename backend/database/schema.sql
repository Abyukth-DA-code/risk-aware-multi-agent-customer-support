-- ============================================================
-- Relational Database Schema for Multi-Agent Customer Support System
-- Version: V4.0 Clean Canonical
-- Domains: E-commerce, Education, Small Business
-- ============================================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    domain VARCHAR(50) NOT NULL CHECK (domain IN ('E-commerce', 'Education', 'Small Business')),
    customer_type VARCHAR(50) NOT NULL,
    account_status VARCHAR(50) NOT NULL,
    verification_status VARCHAR(50) NOT NULL,
    dataset_version VARCHAR(10) NOT NULL DEFAULT 'V4'
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    order_status VARCHAR(50) NOT NULL,
    order_amount_inr NUMERIC(12, 2) NOT NULL CHECK (order_amount_inr >= 0),
    payment_status VARCHAR(50) NOT NULL,
    delivery_status VARCHAR(50) NOT NULL,
    return_status VARCHAR(50) NOT NULL,
    dataset_version VARCHAR(10) NOT NULL DEFAULT 'V4',
    domain VARCHAR(50) NOT NULL DEFAULT 'E-commerce',
    CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    order_id VARCHAR(20) NOT NULL,
    transaction_type VARCHAR(50) NOT NULL,
    amount_inr NUMERIC(12, 2) NOT NULL CHECK (amount_inr >= 0),
    payment_method VARCHAR(50) NOT NULL,
    transaction_status VARCHAR(50) NOT NULL,
    fraud_flag INTEGER NOT NULL CHECK (fraud_flag IN (0, 1)),
    dataset_version VARCHAR(10) NOT NULL DEFAULT 'V4',
    domain VARCHAR(50) NOT NULL DEFAULT 'E-commerce',
    CONSTRAINT fk_tx_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    CONSTRAINT fk_tx_order FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS support_tickets (
    ticket_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    query_type VARCHAR(100) NOT NULL,
    customer_query TEXT NOT NULL,
    ticket_status VARCHAR(50) NOT NULL,
    previous_resolution TEXT,
    dataset_version VARCHAR(10) NOT NULL DEFAULT 'V4',
    domain VARCHAR(50) NOT NULL,
    CONSTRAINT fk_tickets_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS education_records (
    student_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    program VARCHAR(100) NOT NULL,
    year INTEGER NOT NULL CHECK (year BETWEEN 1 AND 5),
    attendance_percent NUMERIC(5, 2) NOT NULL CHECK (attendance_percent BETWEEN 0 AND 100),
    fee_status VARCHAR(50) NOT NULL,
    exam_status VARCHAR(50) NOT NULL,
    scholarship_status VARCHAR(50) NOT NULL DEFAULT 'Not Applied',
    dataset_version VARCHAR(10) NOT NULL DEFAULT 'V4',
    domain VARCHAR(50) NOT NULL DEFAULT 'Education',
    CONSTRAINT fk_students_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS business_records (
    business_record_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    record_type VARCHAR(100) NOT NULL,
    reference_id VARCHAR(50) NOT NULL,
    amount_inr NUMERIC(12, 2) NOT NULL CHECK (amount_inr >= 0),
    record_status VARCHAR(50) NOT NULL,
    approval_status VARCHAR(50) NOT NULL,
    dataset_version VARCHAR(10) NOT NULL DEFAULT 'V4',
    domain VARCHAR(50) NOT NULL DEFAULT 'Small Business',
    CONSTRAINT fk_biz_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE IF NOT EXISTS customer_memory (
    memory_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    related_ticket_id VARCHAR(20),
    memory_type VARCHAR(50) NOT NULL CHECK (memory_type IN ('Short-Term', 'Long-Term')),
    memory_category VARCHAR(100) NOT NULL,
    memory_content TEXT NOT NULL,
    source VARCHAR(50) NOT NULL DEFAULT 'Support_Tickets',
    confidence NUMERIC(4, 3) NOT NULL,
    created_date DATE NOT NULL,
    last_updated DATE NOT NULL,
    validity VARCHAR(50) NOT NULL DEFAULT 'Active',
    privacy_level VARCHAR(50) NOT NULL DEFAULT 'Internal',
    CONSTRAINT fk_mem_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    CONSTRAINT fk_mem_ticket FOREIGN KEY (related_ticket_id) REFERENCES support_tickets(ticket_id)
);

-- Performance Indexes
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_tx_customer ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_tx_order ON transactions(order_id);
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON support_tickets(customer_id);
CREATE INDEX IF NOT EXISTS idx_students_customer ON education_records(customer_id);
CREATE INDEX IF NOT EXISTS idx_biz_customer ON business_records(customer_id);
CREATE INDEX IF NOT EXISTS idx_mem_customer ON customer_memory(customer_id);
