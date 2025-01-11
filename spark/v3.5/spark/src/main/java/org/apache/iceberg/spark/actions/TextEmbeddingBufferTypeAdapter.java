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
package org.apache.iceberg.spark.actions;

import com.google.gson.Gson;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;

public class TextEmbeddingBufferTypeAdapter<T> extends TypeAdapter<TextEmbeddingBuffer> {

  @Override
  public void write(JsonWriter out, TextEmbeddingBuffer value) throws IOException {
    Gson gson = new Gson();
    out.jsonValue(gson.toJson(value));
  }

  @Override
  public TextEmbeddingBuffer read(JsonReader in) throws IOException {
    Gson gson = new Gson();
    return gson.fromJson(in, TextEmbeddingBuffer.class);
  }
}
