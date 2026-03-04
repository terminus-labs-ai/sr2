import time

from sr2.pipeline.result import PipelineResult, StageResult, StageTimer


class TestPipelineResult:
    def test_empty_pipeline_has_success_status(self):
        result = PipelineResult()
        assert result.overall_status == "success"

    def test_failed_stage_makes_overall_failed(self):
        result = PipelineResult()
        result.add_stage(StageResult(stage_name="llm", status="failed", error="timeout"))
        assert result.overall_status == "failed"

    def test_degraded_stage_makes_overall_degraded(self):
        result = PipelineResult()
        result.add_stage(StageResult(stage_name="llm", status="degraded", fallback_used=True))
        assert result.overall_status == "degraded"

    def test_total_tokens_sums_across_stages(self):
        result = PipelineResult()
        result.add_stage(StageResult(stage_name="stage_a", status="success", tokens_used=100))
        result.add_stage(StageResult(stage_name="stage_b", status="success", tokens_used=250))
        result.add_stage(StageResult(stage_name="stage_c", status="success", tokens_used=50))
        assert result.total_tokens == 400


class TestStageTimer:
    def test_timer_measures_nonzero_duration(self):
        with StageTimer("test_stage") as timer:
            time.sleep(0.01)
        assert timer.duration_ms > 0

    def test_timer_result_produces_correct_stage_result(self):
        with StageTimer("summarize") as timer:
            time.sleep(0.01)
        sr = timer.result(status="success", tokens_used=42, fallback_used=False, error=None)
        assert isinstance(sr, StageResult)
        assert sr.stage_name == "summarize"
        assert sr.status == "success"
        assert sr.tokens_used == 42
        assert sr.fallback_used is False
        assert sr.error is None
        assert sr.duration_ms > 0
