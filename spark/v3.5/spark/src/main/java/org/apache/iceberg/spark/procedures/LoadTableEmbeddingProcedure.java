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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import org.apache.iceberg.StatisticsFile;
import org.apache.iceberg.Table;
import org.apache.iceberg.io.InputFile;
import org.apache.iceberg.puffin.Puffin;
import org.apache.iceberg.puffin.PuffinReader;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.spark.actions.TextEmbeddingBuffer;
import org.apache.iceberg.spark.source.SparkTable;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.TableCatalog;
import org.apache.spark.sql.connector.iceberg.catalog.ProcedureParameter;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.types.UTF8String;

public class LoadTableEmbeddingProcedure extends BaseProcedure {

  private static final ProcedureParameter TABLE_PARAM =
      ProcedureParameter.required("table", DataTypes.StringType);
  private static final ProcedureParameter SNAPSHOT_ID_PARAM =
      ProcedureParameter.optional("snapshot_id", DataTypes.LongType);

  private static final ProcedureParameter[] PARAMETERS =
      new ProcedureParameter[] {TABLE_PARAM, SNAPSHOT_ID_PARAM};

  private static final StructType OUTPUT_TYPE =
      new StructType(
          new StructField[] {
            new StructField("embedding_json", DataTypes.StringType, true, Metadata.empty())
          });

  public static SparkProcedures.ProcedureBuilder builder() {
    return new Builder<LoadTableEmbeddingProcedure>() {
      @Override
      protected LoadTableEmbeddingProcedure doBuild() {
        return new LoadTableEmbeddingProcedure(tableCatalog());
      }
    };
  }

  private LoadTableEmbeddingProcedure(TableCatalog tableCatalog) {
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
    SparkTable sparkTable = loadSparkTable(tableIdent);
    Long snapshotId =
        input.asLong(SNAPSHOT_ID_PARAM, sparkTable.table().currentSnapshot().snapshotId());
    List<StatisticsFile> statisticsFiles = sparkTable.table().statisticsFiles();
    for (StatisticsFile statisticsFile : statisticsFiles) {
      if (statisticsFile.snapshotId() == snapshotId) {
        List<TextEmbeddingBuffer> textEmbeddingBuffers =
            getTextEmbeddingBuffers(sparkTable.table(), statisticsFile);
        InternalRow row = newInternalRow(UTF8String.fromString(textEmbeddingBuffers.toString()));
        return new InternalRow[] {row};
      }
    }
    return new InternalRow[0];
  }

  private List<TextEmbeddingBuffer> getTextEmbeddingBuffers(
      Table table, StatisticsFile statisticsFile) {
    InputFile inputFile = table.io().newInputFile(statisticsFile.path());
    List<TextEmbeddingBuffer> textEmbeddingBuffers = Lists.newArrayList();

    try (PuffinReader puffinReader = Puffin.read(inputFile).build()) {
      Iterable<org.apache.iceberg.util.Pair<org.apache.iceberg.puffin.BlobMetadata, ByteBuffer>>
          pairs = puffinReader.readAll(puffinReader.fileMetadata().blobs());
      for (org.apache.iceberg.util.Pair<org.apache.iceberg.puffin.BlobMetadata, ByteBuffer> pair :
          pairs) {
        org.apache.iceberg.puffin.BlobMetadata blobMetadata = pair.first();
        ByteBuffer byteBuffer = pair.second();

        textEmbeddingBuffers.add(TextEmbeddingBuffer.deserialize(byteBuffer.array()));
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return textEmbeddingBuffers;
  }
}
