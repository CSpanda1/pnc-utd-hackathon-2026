"""Offline provider regressions; no inference dependencies needed."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import local_hf
from llm_demo import build_evidence, narrate


class LocalModelTests(unittest.TestCase):
    def tearDown(self):
        local_hf.load_model.cache_clear()

    def test_narration_passes_grounded_evidence(self):
        evidence = build_evidence()
        with patch.object(local_hf, 'generate_local', return_value='local answer') as generate:
            self.assertEqual(narrate(evidence, 'huggingface'), 'local answer')
        prompt = generate.call_args.args[0]
        self.assertIn('synthetic', prompt)
        self.assertIn(str(evidence['scenario']['new_monthly_payment']), prompt)

    def test_unknown_provider_rejected(self):
        with self.assertRaises(ValueError):
            narrate({}, 'typo')

    def test_local_only_architecture_selection_and_reuse(self):
        for model_type, vision in [('qwen3_vl', True), ('qwen2_5_vl', True),
                                   ('qwen3_5', True), ('mistral', False)]:
            with self.subTest(model_type=model_type):
                local_hf.load_model.cache_clear()
                tf = MagicMock()
                tf.AutoConfig.from_pretrained.return_value = SimpleNamespace(model_type=model_type)
                torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                        float32='float32', float16='float16')
                with patch.dict(sys.modules, {'torch': torch, 'transformers': tf}):
                    first = local_hf.load_model('cached/model', '/cache', 'auto', 'auto')
                    self.assertIs(first, local_hf.load_model('cached/model', '/cache', 'auto', 'auto'))
                loader = tf.AutoModelForImageTextToText if vision else tf.AutoModelForCausalLM
                loader.from_pretrained.assert_called_once()
                for call in [tf.AutoConfig.from_pretrained, tf.AutoTokenizer.from_pretrained,
                             loader.from_pretrained]:
                    self.assertTrue(call.call_args.kwargs['local_files_only'])
                    self.assertFalse(call.call_args.kwargs['trust_remote_code'])
                    self.assertEqual(call.call_args.kwargs['cache_dir'], '/cache')
                self.assertEqual(loader.from_pretrained.call_args.kwargs['device_map'], {'': 'cpu'})

    def test_missing_cache_does_not_fall_back_to_download(self):
        tf = MagicMock()
        tf.AutoConfig.from_pretrained.side_effect = OSError('not cached')
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': torch, 'transformers': tf}):
            with self.assertRaisesRegex(RuntimeError, 'no download was attempted'):
                local_hf.load_model('missing/model', None, 'cpu', 'none')
        tf.AutoConfig.from_pretrained.assert_called_once()

    def test_generation_budget_rejected_before_loading(self):
        with patch.dict(os.environ, {'HF_MAX_NEW_TOKENS': '0'}):
            with self.assertRaises(ValueError):
                local_hf.generate_local('evidence')


if __name__ == '__main__':
    unittest.main()
