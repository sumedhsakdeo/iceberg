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
package org.apache.iceberg.spark.procedures;

import org.apache.iceberg.StatisticsFile;
import org.apache.iceberg.actions.ComputeTableEmbeddings;
import org.apache.iceberg.spark.procedures.SparkProcedures.ProcedureBuilder;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.TableCatalog;
import org.apache.spark.sql.connector.iceberg.catalog.ProcedureParameter;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.types.UTF8String;

public class ComputeTableEmbeddingsProcedure extends BaseProcedure {

  private static final ProcedureParameter TABLE_PARAM =
      ProcedureParameter.required("table", DataTypes.StringType);
  private static final ProcedureParameter SNAPSHOT_ID_PARAM =
      ProcedureParameter.optional("snapshot_id", DataTypes.LongType);
  private static final ProcedureParameter COLUMNS_PARAM =
      ProcedureParameter.optional("columns", STRING_ARRAY);
  private static final ProcedureParameter MODEL_NAME_PARAM =
      ProcedureParameter.required("model_name", DataTypes.StringType);
  private static final ProcedureParameter MODEL_INPUTS_PARAM =
      ProcedureParameter.required("model_inputs", STRING_MAP);

  private static final ProcedureParameter[] PARAMETERS =
      new ProcedureParameter[] {
        TABLE_PARAM, SNAPSHOT_ID_PARAM, COLUMNS_PARAM, MODEL_NAME_PARAM, MODEL_INPUTS_PARAM
      };

  private static final StructType OUTPUT_TYPE =
      new StructType(
          new StructField[] {
            new StructField("statistics_file", DataTypes.StringType, true, Metadata.empty())
          });

  public static ProcedureBuilder builder() {
    return new Builder<ComputeTableEmbeddingsProcedure>() {
      @Override
      protected ComputeTableEmbeddingsProcedure doBuild() {
        return new ComputeTableEmbeddingsProcedure(tableCatalog());
      }
    };
  }

  private ComputeTableEmbeddingsProcedure(TableCatalog tableCatalog) {
    super(tableCatalog);
  }

  @Override
  public ProcedureParameter[] parameters() {
    return PARAMETERS;
  }

  @Override
  public StructType outputType() {
    return OUTPUT_TYPE;
  }

  @Override
  public InternalRow[] call(InternalRow args) {
    ProcedureInput input = new ProcedureInput(spark(), tableCatalog(), PARAMETERS, args);
    Identifier tableIdent = input.ident(TABLE_PARAM);
    Long snapshotId = input.asLong(SNAPSHOT_ID_PARAM, null);
    String[] columns = input.asStringArray(COLUMNS_PARAM, null);
    return modifyIcebergTable(
        tableIdent,
        table -> {
          ComputeTableEmbeddings action = actions().computeTableEmbeddings(table);

          if (snapshotId != null) {
            action.snapshot(snapshotId);
          }

          if (columns != null) {
            action.columns(columns);
          }

          ComputeTableEmbeddings.Result result = action.execute();
          return toOutputRows(result);
        });
  }

  private InternalRow[] toOutputRows(ComputeTableEmbeddings.Result result) {
    StatisticsFile statisticsFile = result.statisticsFile();
    if (statisticsFile != null) {
      InternalRow row = newInternalRow(UTF8String.fromString(statisticsFile.path()));
      return new InternalRow[] {row};
    } else {
      return new InternalRow[0];
    }
  }
}
