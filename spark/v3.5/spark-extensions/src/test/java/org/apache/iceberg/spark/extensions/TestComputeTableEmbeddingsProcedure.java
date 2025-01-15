/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.spark.extensions;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

import org.apache.iceberg.ParameterizedTestExtension;
import org.apache.iceberg.Table;
import org.apache.iceberg.spark.actions.EmbeddingModelBuilder;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.TestTemplate;
import org.junit.jupiter.api.extension.ExtendWith;

@ExtendWith(ParameterizedTestExtension.class)
public class TestComputeTableEmbeddingsProcedure extends ExtensionsTestBase {

  @AfterEach
  public void removeTables() {
    sql("DROP TABLE IF EXISTS %s", tableName);
    sql("DROP TABLE IF EXISTS %s_BACKUP_", tableName);
  }

  @TestTemplate
  public void testComputeTableEmbeddings() {
    assumeThat(catalogName).isEqualToIgnoringCase("spark_catalog");

    String text = "hello";
    sql("CREATE TABLE %s (id int, data string) USING iceberg", tableName);
    sql("INSERT into %s values(1, '%s')", tableName, text);

    Table table = validationCatalog.loadTable(tableIdent);

    assertThat(table.statisticsFiles()).isNull();

    sql(
        "CALL %s.system.compute_table_embeddings ("
            + "table => '%s', "
            + "model_name => '%s', "
            + "model_inputs => map('x', 'y'), "
            + "snapshot_id => %s, "
            + "columns => array('data') "
            + ")",
        catalogName,
        tableName,
        EmbeddingModelBuilder.ALL_MINI_LM_L6V2,
        table.currentSnapshot().snapshotId());
    table.refresh();

    assertThat(table.statisticsFiles()).isNotNull();
  }
}
