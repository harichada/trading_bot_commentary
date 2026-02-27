import React, {useState} from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import Animated, {FadeInDown, FadeIn} from 'react-native-reanimated';
import {Ionicons} from '@expo/vector-icons';
import {HapticButton} from '../components/HapticButton';
import {colors, typography, spacing, borderRadius} from '../theme';
import Toast from 'react-native-toast-message';

interface Props {
  onConnect: (url: string, apiKey: string) => void;
}

export function ServerSetupScreen({onConnect}: Props) {
  const [url, setUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [testing, setTesting] = useState(false);
  const [connected, setConnected] = useState<boolean | null>(null);

  const testAndConnect = async () => {
    const trimmed = url.replace(/\/+$/, '');
    if (!trimmed) {
      Toast.show({type: 'error', text1: 'Enter a server URL'});
      return;
    }
    if (!apiKey.trim()) {
      Toast.show({type: 'error', text1: 'Enter your API key'});
      return;
    }

    setTesting(true);
    setConnected(null);
    try {
      const headers: Record<string, string> = {'Content-Type': 'application/json'};
      if (apiKey) headers['X-API-Key'] = apiKey;

      const res = await fetch(`${trimmed}/api/health`, {headers});
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setConnected(true);
      onConnect(trimmed, apiKey);
    } catch (err: any) {
      setConnected(false);
      Toast.show({type: 'error', text1: 'Connection Failed', text2: err.message});
    } finally {
      setTesting(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <Animated.View entering={FadeIn.duration(600)} style={styles.card}>
          <Animated.View entering={FadeInDown.delay(100).duration(400)}>
            <Ionicons name="server" size={48} color={colors.cyan} style={styles.logo} />
            <Text style={styles.title}>Gap Fade</Text>
            <Text style={styles.subtitle}>Connect to your trading server</Text>
          </Animated.View>

          <Animated.View entering={FadeInDown.delay(200).duration(300)}>
            <View style={styles.field}>
              <Text style={styles.label}>SERVER URL</Text>
              <View style={styles.inputWrapper}>
                <Ionicons name="globe-outline" size={18} color={colors.textMuted} />
                <TextInput
                  style={styles.input}
                  placeholder="http://192.168.1.100:8002"
                  placeholderTextColor={colors.textMuted}
                  value={url}
                  onChangeText={setUrl}
                  autoCapitalize="none"
                  autoCorrect={false}
                  keyboardType="url"
                />
                {connected !== null && (
                  <Ionicons
                    name={connected ? 'checkmark-circle' : 'close-circle'}
                    size={20}
                    color={connected ? colors.green : colors.red}
                  />
                )}
              </View>
            </View>
          </Animated.View>

          <Animated.View entering={FadeInDown.delay(300).duration(300)}>
            <View style={styles.field}>
              <Text style={styles.label}>API KEY</Text>
              <View style={styles.inputWrapper}>
                <Ionicons name="key-outline" size={18} color={colors.textMuted} />
                <TextInput
                  style={styles.input}
                  placeholder="Enter your API key"
                  placeholderTextColor={colors.textMuted}
                  value={apiKey}
                  onChangeText={setApiKey}
                  autoCapitalize="none"
                  autoCorrect={false}
                  secureTextEntry
                />
              </View>
            </View>
          </Animated.View>

          <Animated.View entering={FadeInDown.delay(400).duration(300)}>
            <HapticButton
              label={testing ? 'Connecting...' : 'Connect'}
              icon="link"
              onPress={testAndConnect}
              disabled={testing}
              loading={testing}
              size="lg"
              style={{marginTop: spacing.sm}}
            />
          </Animated.View>
        </Animated.View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  scroll: {flexGrow: 1, justifyContent: 'center', padding: spacing.xl},
  card: {
    backgroundColor: colors.bgCard, borderRadius: borderRadius.xl,
    padding: spacing.xl, borderWidth: 1, borderColor: colors.border,
  },
  logo: {textAlign: 'center', marginBottom: spacing.md},
  title: {...typography.h1, color: colors.cyan, textAlign: 'center', marginBottom: spacing.xs},
  subtitle: {...typography.body, color: colors.textSecondary, textAlign: 'center', marginBottom: spacing.xl},
  field: {marginBottom: spacing.lg},
  label: {...typography.label, color: colors.textMuted, marginBottom: spacing.xs},
  inputWrapper: {
    flexDirection: 'row', alignItems: 'center', gap: spacing.sm,
    backgroundColor: colors.bgInput, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.lg, paddingHorizontal: spacing.md,
  },
  input: {
    flex: 1, ...typography.mono, color: colors.textPrimary,
    paddingVertical: spacing.sm + 2,
  },
});
