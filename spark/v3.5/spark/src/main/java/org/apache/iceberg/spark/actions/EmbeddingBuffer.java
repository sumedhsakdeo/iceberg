/*
 *
 *  * Licensed to the Apache Software Foundation (ASF) under one
 *  * or more contributor license agreements.  See the NOTICE file
 *  * distributed with this work for additional information
 *  * regarding copyright ownership.  The ASF licenses this file
 *  * to you under the Apache License, Version 2.0 (the
 *  * "License"); you may not use this file except in compliance
 *  * with the License.  You may obtain a copy of the License at
 *  *
 *  *   http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  * Unless required by applicable law or agreed to in writing,
 *  * software distributed under the License is distributed on an
 *  * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  * KIND, either express or implied.  See the License for the
 *  * specific language governing permissions and limitations
 *  * under the License.
 *
 */

package org.apache.iceberg.spark.actions;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;

public class EmbeddingBuffer implements Serializable {

    public static final Encoder<EmbeddingBuffer> ENCODER = Encoders.bean(EmbeddingBuffer.class);

    private static final long serialVersionUID = 1L;

    private List<Embedding> embeddingList;

    public EmbeddingBuffer() {
        embeddingList = new ArrayList<>();
    }

    public EmbeddingBuffer(List<Embedding> embeddingList) {
        this.embeddingList = embeddingList;
    }

    public List<Embedding> getEmbeddingList() {
        return embeddingList;
    }

    public void setEmbeddingList(List<Embedding> embeddingList) {
        this.embeddingList = embeddingList;
    }

    public void addEmbedding(Embedding embedding) {
        embeddingList.add(embedding);
    }

    public byte[] serialize() {
        try(ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutput out = new ObjectOutputStream(bos)) {
            out.writeObject(this);
            return bos.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException("Failed to serialize EmbeddingBuffer", e);
        }
    }

    public static EmbeddingBuffer deserialize(byte[] bytes) {
        try (ByteArrayInputStream byteStream = new ByteArrayInputStream(bytes);
             ObjectInputStream objectStream = new ObjectInputStream(byteStream)) {
            return (EmbeddingBuffer) objectStream.readObject(); // Deserialize the object
        } catch (IOException | ClassNotFoundException e) {
            throw new RuntimeException("Error deserializing EmbeddingBuffer", e);
        }
    }

}
