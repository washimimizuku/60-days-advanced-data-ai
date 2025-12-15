-- Day 11: Sample Row-Level Security Policies
-- Demonstrates various RLS patterns for access control

-- Enable RLS on all tables
ALTER TABLE customer_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE financial_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE employee_data ENABLE ROW LEVEL SECURITY;

-- Create application roles
DO $$
BEGIN
    -- Create roles if they don't exist
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'data_analyst') THEN
        CREATE ROLE data_analyst;
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'data_scientist') THEN
        CREATE ROLE data_scientist;
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'data_engineer') THEN
        CREATE ROLE data_engineer;
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'manager') THEN
        CREATE ROLE manager;
    END IF;
    
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'admin_role') THEN
        CREATE ROLE admin_role;
    END IF;
END
$$;

-- Grant basic permissions to roles
GRANT SELECT ON customer_data TO data_analyst, data_scientist, data_engineer, manager;
GRANT SELECT ON financial_data TO data_scientist, manager;
GRANT SELECT, INSERT, UPDATE ON customer_data TO data_engineer;
GRANT SELECT ON employee_data TO manager;
GRANT ALL ON ALL TABLES IN SCHEMA public TO admin_role;

-- Function to get current user's tenant context
CREATE OR REPLACE FUNCTION get_current_tenant_id()
RETURNS UUID AS $$
BEGIN
    -- In a real application, this would get the tenant from session context
    -- For demo purposes, we'll use a session variable
    RETURN COALESCE(
        current_setting('app.current_tenant_id', true)::UUID,
        '550e8400-e29b-41d4-a716-446655440001'::UUID  -- Default to Acme Corp
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get current user's regions
CREATE OR REPLACE FUNCTION get_user_regions()
RETURNS TEXT[] AS $$
BEGIN
    -- In production, this would query user permissions
    RETURN COALESCE(
        string_to_array(current_setting('app.user_regions', true), ','),
        ARRAY['US']  -- Default regions
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get current user's clearance level
CREATE OR REPLACE FUNCTION get_user_clearance_level()
RETURNS INTEGER AS $$
BEGIN
    RETURN COALESCE(
        current_setting('app.user_clearance_level', true)::INTEGER,
        1  -- Default clearance level
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check business hours
CREATE OR REPLACE FUNCTION is_business_hours()
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXTRACT(hour FROM CURRENT_TIME) BETWEEN 9 AND 17
           AND EXTRACT(dow FROM CURRENT_DATE) BETWEEN 1 AND 5;
END;
$$ LANGUAGE plpgsql;

-- RLS Policy 1: Tenant Isolation (applies to all roles)
CREATE POLICY tenant_isolation_customer ON customer_data
    FOR ALL
    TO public
    USING (tenant_id = get_current_tenant_id());

CREATE POLICY tenant_isolation_financial ON financial_data
    FOR ALL
    TO public
    USING (tenant_id = get_current_tenant_id());

CREATE POLICY tenant_isolation_employee ON employee_data
    FOR ALL
    TO public
    USING (tenant_id = get_current_tenant_id());

-- RLS Policy 2: Regional Access Control
CREATE POLICY regional_access_customer ON customer_data
    FOR SELECT
    TO data_analyst, data_scientist
    USING (region = ANY(get_user_regions()));

-- RLS Policy 3: Sensitivity-Based Access Control
CREATE POLICY sensitivity_filter_customer ON customer_data
    FOR SELECT
    TO data_analyst
    USING (
        CASE sensitivity_level
            WHEN 'public' THEN true
            WHEN 'internal' THEN get_user_clearance_level() >= 2
            WHEN 'confidential' THEN get_user_clearance_level() >= 3
            WHEN 'secret' THEN get_user_clearance_level() >= 4
            ELSE false
        END
    );

CREATE POLICY sensitivity_filter_financial ON financial_data
    FOR SELECT
    TO data_scientist, manager
    USING (get_user_clearance_level() >= 3);

-- RLS Policy 4: Hierarchical Access (Managers see subordinate data)
CREATE POLICY manager_hierarchy_employee ON employee_data
    FOR SELECT
    TO manager
    USING (
        manager_id = current_setting('app.user_employee_id', true)::INTEGER
        OR employee_id = current_setting('app.user_employee_id', true)::INTEGER
    );

-- RLS Policy 5: Time-Based Access Control
CREATE POLICY business_hours_financial ON financial_data
    FOR SELECT
    TO data_analyst
    USING (is_business_hours());

-- RLS Policy 6: Department-Based Access Control
CREATE POLICY department_access_customer ON customer_data
    FOR SELECT
    TO data_analyst
    USING (
        department = current_setting('app.user_department', true)
        OR current_setting('app.user_department', true) = 'Analytics'  -- Analytics can see all
    );

-- RLS Policy 7: Write Restrictions (Data Engineers can only modify non-confidential data)
CREATE POLICY write_restriction_customer ON customer_data
    FOR UPDATE
    TO data_engineer
    USING (sensitivity_level != 'confidential')
    WITH CHECK (sensitivity_level != 'confidential');

CREATE POLICY insert_restriction_customer ON customer_data
    FOR INSERT
    TO data_engineer
    WITH CHECK (sensitivity_level != 'confidential');

-- RLS Policy 8: Admin Override (Admins bypass all restrictions)
CREATE POLICY admin_override_customer ON customer_data
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY admin_override_financial ON financial_data
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY admin_override_employee ON employee_data
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

-- Create view for secure customer access
CREATE OR REPLACE VIEW secure_customer_view AS
SELECT 
    id,
    customer_name,
    CASE 
        WHEN get_user_clearance_level() >= 2 THEN email
        ELSE 'REDACTED'
    END as email,
    CASE 
        WHEN get_user_clearance_level() >= 2 THEN phone
        ELSE 'REDACTED'
    END as phone,
    region,
    department,
    CASE 
        WHEN get_user_clearance_level() >= 3 THEN sensitivity_level
        ELSE 'CLASSIFIED'
    END as sensitivity_level,
    created_at
FROM customer_data;

-- Grant access to the secure view
GRANT SELECT ON secure_customer_view TO data_analyst, data_scientist, data_engineer, manager;

-- Create audit trigger function
CREATE OR REPLACE FUNCTION audit_access_log()
RETURNS TRIGGER AS $$
BEGIN
    -- Log data access for audit purposes
    INSERT INTO audit_log (
        event_id,
        event_type,
        user_id,
        tenant_id,
        resource,
        action,
        result,
        additional_data
    ) VALUES (
        uuid_generate_v4()::TEXT,
        'data_access',
        current_user,
        get_current_tenant_id(),
        TG_TABLE_NAME,
        TG_OP,
        'success',
        jsonb_build_object(
            'table', TG_TABLE_NAME,
            'operation', TG_OP,
            'timestamp', CURRENT_TIMESTAMP
        )
    );
    
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers
CREATE TRIGGER customer_data_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON customer_data
    FOR EACH ROW EXECUTE FUNCTION audit_access_log();

CREATE TRIGGER financial_data_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON financial_data
    FOR EACH ROW EXECUTE FUNCTION audit_access_log();

CREATE TRIGGER employee_data_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON employee_data
    FOR EACH ROW EXECUTE FUNCTION audit_access_log();

-- Example of setting session context (would be done by application)
-- SELECT set_config('app.current_tenant_id', '550e8400-e29b-41d4-a716-446655440001', false);
-- SELECT set_config('app.user_regions', 'US,CA', false);
-- SELECT set_config('app.user_clearance_level', '2', false);
-- SELECT set_config('app.user_department', 'Analytics', false);
-- SELECT set_config('app.user_employee_id', '1001', false);

-- Test queries (commented out for initialization)
/*
-- Test tenant isolation
SET app.current_tenant_id = '550e8400-e29b-41d4-a716-446655440001';
SELECT * FROM customer_data;  -- Should only see Acme Corp data

-- Test regional access
SET app.user_regions = 'US';
SELECT * FROM customer_data;  -- Should only see US data

-- Test sensitivity filtering
SET app.user_clearance_level = '2';
SELECT * FROM customer_data;  -- Should see public and internal data

-- Test secure view
SELECT * FROM secure_customer_view;  -- Should see redacted data based on clearance
*/