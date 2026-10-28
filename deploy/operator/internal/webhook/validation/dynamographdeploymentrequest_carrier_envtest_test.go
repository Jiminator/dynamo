package validation_test

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
)

// A request created through its owning API version has to stay writable through every other
// served version. A metadata-only patch through the foreign view must not be rejected, and it
// must neither rewrite the spec nor bump the generation. A default that materialises a native
// field next to a conversion carrier breaks this, so it is covered at the API-server boundary.
func TestDynamoGraphDeploymentRequestCarrierObjectsStayWritable(t *testing.T) {
	t.Run("native v1beta1 written through v1beta2", func(t *testing.T) {
		carrierRoundTrip(t, betaDGDRForAdmission(nil), beta2DGDRForAdmission(nil), "v1beta1")
	})
	t.Run("native v1beta2 written through v1beta1", func(t *testing.T) {
		carrierRoundTrip(t, beta2DGDRForAdmission(setBeta2AdmissionHardware), betaDGDRForAdmission(nil), "v1beta2")
	})
	t.Run("native v1beta2 written through v1alpha1", func(t *testing.T) {
		carrierRoundTrip(t, beta2DGDRForAdmission(setBeta2AdmissionHardware), alphaDGDRForAdmission(nil), "v1beta2")
	})
	t.Run("native v1beta2 trace written through v1beta1", func(t *testing.T) {
		foreign := carrierRoundTrip(t, beta2DGDRForAdmission(setBeta2AdmissionTrace), betaDGDRForAdmission(nil), "v1beta2")

		t.Log("The v1beta1 carrier holds the complete trace and no static default")
		trace, found, err := unstructured.NestedMap(foreign.Object, "spec", "v1beta2", "workload", "trace")
		if err != nil || !found {
			t.Fatalf("spec.v1beta2.workload.trace found=%v err=%v", found, err)
		}
		path, _, _ := unstructured.NestedString(trace, "source", "pvc", "path")
		if trace["format"] != "dynamo" || path != "retune/w-1.jsonl.gz" {
			t.Fatalf("spec.v1beta2.workload.trace = %#v", trace)
		}
		if _, found, _ := unstructured.NestedFieldNoCopy(foreign.Object, "spec", "v1beta2", "workload", "static"); found {
			t.Fatal("the carrier gained a static workload beside the trace")
		}
	})
}

// A v1beta1 carrier has no trace schema, so a carrier seeded without admission can hold a trace
// path the v1beta2 CEL rules reject. CEL ratcheting then admits every write that leaves that path
// unchanged, which leaves the v1beta2 validating webhook as the only check on the next write.
func TestDynamoGraphDeploymentRequestCarrierTraceIsValidatedOnTheNextWrite(t *testing.T) {
	env := admissionEnv.ForTest(t)
	admissionGate.set(features.Gates{GPUDiscovery: true})
	setAdmissionH100Topology()
	warnings := &warningRecorder{}
	seeder := newAdmissionResourceClient(t, env, betaDGDRForAdmission(nil), legacySeedUsername, warnings)
	unadmittedBeta2 := newAdmissionResourceClient(t, env, beta2DGDRForAdmission(nil), legacySeedUsername, warnings)
	beta2 := newAdmissionResourceClient(t, env, beta2DGDRForAdmission(nil), "", warnings)

	t.Log("Seed a v1beta1 carrier whose trace path leaves the claim, without admission")
	carrier := &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1beta1",
		"kind":       "DynamoGraphDeploymentRequest",
		"metadata":   map[string]any{"name": "seeded-trace", "namespace": env.Namespace()},
		"spec": map[string]any{"v1beta2": map[string]any{
			"model":    map[string]any{"name": "Qwen/Qwen3-0.6B"},
			"hardware": map[string]any{"gpu": map[string]any{"sku": "h100_sxm", "budget": int64(4)}},
			"workload": map[string]any{"trace": map[string]any{
				"format": "dynamo",
				"source": map[string]any{"pvc": map[string]any{"claimName": "request-traces", "path": "../escape.jsonl"}},
			}},
		}},
	}}
	if _, err := seeder.Create(t.Context(), carrier, metav1.CreateOptions{}); err != nil {
		t.Fatalf("seed the carrier without admission: %v", err)
	}

	t.Log("The schema alone admits a v1beta2 write that keeps the invalid path")
	if _, err := unadmittedBeta2.Patch(t.Context(), "seeded-trace", types.MergePatchType,
		[]byte(`{"metadata":{"labels":{"trace-probe-unadmitted":"1"}}}`), metav1.PatchOptions{}); err != nil {
		t.Fatalf("metadata patch without admission: %v", err)
	}

	t.Log("The validating webhook rejects the same write")
	_, err := beta2.Patch(t.Context(), "seeded-trace", types.MergePatchType,
		[]byte(`{"metadata":{"labels":{"trace-probe-admitted":"1"}}}`), metav1.PatchOptions{})
	const want = `spec.workload.trace.source.pvc.path: Invalid value: "../escape.jsonl": path must be relative to the claim root`
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("metadata patch through admission: error = %v, want %q", err, want)
	}
}

func setBeta2AdmissionTrace(request *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	setBeta2AdmissionHardware(request)
	request.Spec.Workload = &nvidiacomv1beta2.WorkloadSpec{Trace: &nvidiacomv1beta2.TraceWorkloadSpec{
		Format: nvidiacomv1beta2.TraceFormatDynamo,
		Source: nvidiacomv1beta2.TraceSourceSpec{PVC: &nvidiacomv1beta2.TracePVCSource{
			ClaimName: "request-traces", Path: "retune/w-1.jsonl.gz",
		}},
	}}
}

func carrierRoundTrip(t *testing.T, ownerFixture, otherFixture runtime.Object, carrierKey string) *unstructured.Unstructured {
	t.Helper()
	env := admissionEnv.ForTest(t)
	admissionGate.set(features.Gates{GPUDiscovery: true})
	setAdmissionH100Topology()
	warnings := &warningRecorder{}
	ownerClient := newAdmissionResourceClient(t, env, ownerFixture, "", warnings)
	otherClient := newAdmissionResourceClient(t, env, otherFixture, "", warnings)

	t.Log("Create the request through the version that owns its representation")
	object, _ := admissionObject(t, ownerFixture, env.Namespace(), nil)
	created, err := ownerClient.Create(t.Context(), object, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create through the owning version: %v", err)
	}
	name := created.GetName()

	t.Log("Read it through the other version and expect the conversion carrier")
	foreign, err := otherClient.Get(t.Context(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("read through the other version: %v", err)
	}
	spec, _ := foreign.Object["spec"].(map[string]any)
	if _, ok := spec[carrierKey]; !ok {
		keys := make([]string, 0, len(spec))
		for key := range spec {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		t.Fatalf("foreign view has spec keys %v, want the %s carrier", keys, carrierKey)
	}

	t.Log("Patch metadata through each version")
	for label, client := range map[string]dynamic.ResourceInterface{"foreign": otherClient, "owner": ownerClient} {
		patch := []byte(`{"metadata":{"labels":{"carrier-probe-` + label + `":"1"}}}`)
		if _, err := client.Patch(t.Context(), name, types.MergePatchType, patch, metav1.PatchOptions{}); err != nil {
			t.Fatalf("metadata-only patch through the %s version: %v", label, err)
		}
	}

	t.Log("The spec and generation must be untouched")
	after, err := ownerClient.Get(t.Context(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("read back through the owning version: %v", err)
	}
	if after.GetGeneration() != created.GetGeneration() {
		t.Fatalf("generation = %d, want %d: a metadata-only patch rewrote the spec", after.GetGeneration(), created.GetGeneration())
	}

	t.Log("Write the unchanged foreign view back with a full update; the spec must be byte-stable")
	beforeSpec, err := json.Marshal(foreign.Object["spec"])
	if err != nil {
		t.Fatalf("encode the foreign spec: %v", err)
	}
	current, err := otherClient.Get(t.Context(), name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("read the foreign view before the update: %v", err)
	}
	updated, err := otherClient.Update(t.Context(), current, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("full update of the unchanged foreign view: %v", err)
	}
	afterSpec, err := json.Marshal(updated.Object["spec"])
	if err != nil {
		t.Fatalf("encode the updated spec: %v", err)
	}
	if !bytes.Equal(beforeSpec, afterSpec) {
		t.Fatalf("the foreign spec changed across an unchanged update:\nbefore: %s\nafter:  %s", beforeSpec, afterSpec)
	}
	if updated.GetGeneration() != created.GetGeneration() {
		t.Fatalf("generation = %d, want %d: an unchanged update rewrote the spec", updated.GetGeneration(), created.GetGeneration())
	}
	return foreign
}
